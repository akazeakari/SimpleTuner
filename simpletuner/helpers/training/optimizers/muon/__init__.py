"""
Different versions appeared,
they have identical interface, but sutiable for different scenarios.
"""

__version__ = "0.1.0"

__all__ = ["MuonClip"]

"""
This implementation uses torch.compile to speed up,
should be suitable for different backends.
"""

import torch
from typing import Optional, Dict
from torch import Tensor
import math

_generators: Dict[torch.device, torch.Generator] = {}

# Constants from PyTorch Muon implementation
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5

def _zeropower_via_newtonschulz(
    grad: Tensor, 
    ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    use_cans: bool = False,
    cans_a_bound: float = 1e-4
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    Optionally supports CANS (Chebyshev-accelerated Newton-Schulz) for faster convergence.
    
    Optimizations: maximized inplace operations, preallocated buffers, fused operations.
    
    Args:
        grad: Input gradient tensor (must be 2D)
        ns_coefficients: Coefficients (a, b, c) for Newton-Schulz polynomial (unused if use_cans=True)
        ns_steps: Number of Newton-Schulz iteration steps
        eps: Term added to denominator for numerical stability
        use_cans: If True, use Chebyshev-accelerated Newton-Schulz for faster convergence
        cans_a_bound: Initial lower bound for singular values when using CANS
    
    Returns:
        Orthogonalized gradient tensor
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if not use_cans and len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    
    transposed = grad.size(-2) > grad.size(-1)
    X = grad.mT if transposed else grad
    
    if grad.dtype != torch.bfloat16:
        X = X.to(torch.bfloat16)
    else:
        X = X.clone()
    
    X.div_(X.norm(dim=(-2, -1), keepdim=True).clamp_(min=eps))
    
    if use_cans:
        lower_bound = cans_a_bound
        upper_bound = 1.0
        inv_3 = 1.0 / 3.0
        
        n = X.size(0)
        A = torch.empty(n, n, dtype=X.dtype, device=X.device)
        AX = torch.empty_like(X)
        
        for _ in range(ns_steps):
            a_bound, b_bound = lower_bound, upper_bound
            
            a_sq = a_bound * a_bound
            b_sq = b_bound * b_bound
            ab = a_bound * b_bound
            
            e_sq = (a_sq + ab + b_sq) * inv_3
            e_pow_1_5 = e_sq * math.sqrt(e_sq)
            
            common_den_part = 2.0 * e_pow_1_5
            ab_part = a_sq * b_bound + b_sq * a_bound
            alpha_den = common_den_part + ab_part
            alpha = 6.0 / alpha_den
            
            c1 = alpha * e_sq
            c3 = -alpha * inv_3
            
            torch.mm(X, X.mT, out=A)
            torch.mm(A, X, out=AX)
            X.mul_(c1).add_(AX, alpha=c3)
            
            eps_val = (common_den_part - ab_part) / alpha_den
            lower_bound = 1.0 - eps_val
            upper_bound = 1.0 + eps_val
    else:
        a, b, c = ns_coefficients
        
        n = X.size(0)
        A = torch.empty(n, n, dtype=X.dtype, device=X.device)
        B = torch.empty(n, n, dtype=X.dtype, device=X.device)
        
        for _ in range(ns_steps):
            torch.mm(X, X.mT, out=A)
            torch.addmm(A, A, A, beta=b, alpha=c, out=B)
            torch.addmm(X, B, X, beta=a, out=X)
    
    if transposed:
        X = X.mT
    
    return X.to(grad.dtype)


def _unnmf(row_col: tuple, out=None) -> torch.Tensor:
    if out is not None:
        return torch.outer(row_col[0], row_col[1], out=out)
    return torch.outer(row_col[0], row_col[1])


def _nnmf(matrix: torch.Tensor, out: tuple):
    shape = matrix.shape
    torch.sum(matrix, dim=1, out=out[0])
    torch.sum(matrix, dim=0, out=out[1])
    
    if shape[0] < shape[1]:
        scale = out[0].sum()
        if scale != 0:
            out[0].div_(scale)
    else:
        scale = out[1].sum()
        if scale != 0:
            out[1].div_(scale)


# Pre-compute bit masks for packing/unpacking optimization
_BIT_MASKS = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)

@torch.no_grad()
def _pack_bools(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack boolean tensor into uint8 tensor (8 bools per uint8).
    
    Optimized implementation using pre-computed bit masks and efficient reshaping.
    
    Args:
        tensor: Boolean tensor of shape [n, m]
    
    Returns:
        Packed uint8 tensor of shape [n, packed_m] where packed_m = (m + 7) // 8
    """
    n, m = tensor.shape
    packed_m = (m + 7) // 8
    
    # Convert to uint8 and pad if necessary
    if m % 8 != 0:
        padded_tensor = torch.nn.functional.pad(tensor, (0, packed_m * 8 - m), 'constant', 0).to(torch.uint8)
    else:
        padded_tensor = tensor.to(torch.uint8)
    
    # Use pre-computed bit masks for packing
    bit_masks = _BIT_MASKS.to(tensor.device)
    reshaped = padded_tensor.view(n, packed_m, 8)
    packed = (reshaped * bit_masks).sum(dim=2, dtype=torch.uint8)
    
    return packed


@torch.no_grad()
def _unpack_bools(packed_tensor: torch.Tensor, original_m: int) -> torch.Tensor:
    """
    Unpack uint8 tensor back to boolean tensor.
    
    Optimized implementation using pre-computed bit masks and efficient bitwise operations.
    
    Args:
        packed_tensor: Packed uint8 tensor of shape [n, packed_m]
        original_m: Original number of columns before packing
    
    Returns:
        Unpacked boolean tensor of shape [n, original_m]
    """
    n, packed_m = packed_tensor.shape
    
    # Use pre-computed bit masks for unpacking
    bit_masks = _BIT_MASKS.to(packed_tensor.device).view(1, 1, 8)
    
    # Expand and apply bitwise AND, then reshape
    unpacked = ((packed_tensor.unsqueeze(2) & bit_masks) != 0).view(n, -1)[:, :original_m]
    
    return unpacked


def _get_effective_shape(numel: int) -> tuple[int, int]:
    if numel <= 0:
        return (0, 0)
    for i in reversed(range(1, int(numel ** 0.5) + 1)):
        if numel % i == 0:
            return (numel // i, i)
    return (numel, 1)


def set_seed(device: torch.device):
    """
    Initializes or resets the deterministic generator for a specific device.
    This ensures that the sequence of random numbers used for stochastic
    rounding is reproducible.
    """
    global _generators
    if device not in _generators:
        _generators[device] = torch.Generator(device=device)
    _generators[device].manual_seed(42)

def copy_stochastic_(target: Tensor, source: Tensor):
    global _generators
    device = source.device
    if device not in _generators:
        set_seed(device)
    
    generator = _generators[device]
    
    result = torch.randint(
        size=source.shape,
        device=source.device,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
        generator=generator,
    )
    
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def add_stochastic_(input: Tensor, other: Tensor, alpha: float = 1.0):
    if other.dtype == torch.float32:
        result = other.add(input, alpha=alpha)
    else:
        result = other.to(dtype=torch.float32).add_(input, alpha=alpha)
    copy_stochastic_(input, result)


class MuonClip(torch.optim.Optimizer):
    """
    MuonClip optimizer as described in Kimi K2 paper.
    
    Combines the token-efficient Muon optimizer with QK-Clip for stability.
    Uses PyTorch's optimized Newton-Schulz iteration with quintic polynomial.
    Optionally supports CANS for faster convergence.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 2e-4)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.1)
        qk_clip_threshold: Maximum attention logit threshold τ (default: 100.0)
        qk_clip_alpha: Balance parameter for Q/K scaling (default: 0.5)
        ns_steps: Number of Newton-Schulz iteration steps (default: 5)
        ns_coefficients: Coefficients (a, b, c) for Newton-Schulz polynomial (default: PyTorch optimal)
        eps: Term added to denominator for numerical stability (default: 1e-7)
        rms_scale_factor: RMS scaling factor (default: 0.2)
        use_smmf: Enable SMMF for memory-efficient momentum storage (default: False)
        vector_reshape: Reshape 1D vectors to 2D for SMMF compression (default: False)
        stochastic_rounding: Use stochastic rounding for BF16 parameters (default: True)
        use_cans: Enable Chebyshev-accelerated Newton-Schulz for faster convergence (default: False)
        cans_a_bound: Initial lower bound for singular values when using CANS (default: 1e-4)
    """
    
    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        qk_clip_threshold: float = 100.0,
        qk_clip_alpha: float = 0.5,
        ns_steps: int = DEFAULT_NS_STEPS,
        ns_coefficients: tuple[float, float, float] = (DEFAULT_A, DEFAULT_B, DEFAULT_C),
        eps: float = EPS,
        rms_scale_factor: float = 0.2,
        use_smmf: bool = False,
        vector_reshape: bool = False,
        stochastic_rounding: bool = True,
        use_cans: bool = False,
        cans_a_bound: float = 1e-4,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if ns_steps >= 100:
            raise ValueError("Number of steps must be less than 100 for computational efficiency")
        if not use_cans and len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must be a tuple of exactly 3 values")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            qk_clip_threshold=qk_clip_threshold,
            qk_clip_alpha=qk_clip_alpha,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            rms_scale_factor=rms_scale_factor,
            use_smmf=use_smmf,
            vector_reshape=vector_reshape,
            use_cans=use_cans,
            cans_a_bound=cans_a_bound,
        )
        super().__init__(params, defaults)
        
        self.stochastic_rounding = stochastic_rounding
        
        if self.stochastic_rounding:
            devices = {p.device for group in self.param_groups for p in group['params'] if p.dtype == torch.bfloat16}
            for device in devices:
                set_seed(device)
    
    @torch.no_grad()
    def step(self, closure=None, attention_max_logits: Optional[Dict[str, torch.Tensor]] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            attention_max_logits: Dict mapping layer names to max logit values per head
                                 Format: {"layer.0.attn": tensor([max_logit_head_0, ...])}
        
        Returns:
            Optional loss from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            ns_coefficients = group['ns_coefficients']
            eps = group['eps']
            rms_scale = group['rms_scale_factor']
            use_smmf = group['use_smmf']
            vector_reshape = group['vector_reshape']
            use_cans = group['use_cans']
            cans_a_bound = group['cans_a_bound']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                use_factored = use_smmf and (grad.dim() >= 2 or (grad.dim() == 1 and vector_reshape))
                
                # State initialization
                if len(state) == 0:
                    if use_factored:
                        if grad.dim() == 1:
                            state['effective_shape'] = _get_effective_shape(grad.numel())
                        else:
                            state['effective_shape'] = (grad.shape[0], grad[0].numel())
                        
                        d1, d2 = state['effective_shape']
                        state['mu_row'] = torch.zeros(d1, device=p.device, dtype=torch.float32)
                        state['mu_col'] = torch.zeros(d2, device=p.device, dtype=torch.float32)
                        packed_d2 = (d2 + 7) // 8
                        state['sign_buf'] = torch.zeros((d1, packed_d2), dtype=torch.uint8, device=p.device)
                        state['mt_buf'] = torch.empty((d1, d2), device=p.device, dtype=torch.float32)
                        state['factored'] = True
                    else:
                        state['momentum_buffer'] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        state['factored'] = False
                
                # 2D parameters or high-dimensional parameters (with orthogonalization)
                if grad.dim() == 2 or (grad.dim() > 2 and use_factored):
                    original_shape = grad.shape
                    
                    # Update momentum buffer
                    if state.get('factored', False):
                        d1, d2 = state['effective_shape']
                        mt_buf = state['mt_buf']
                        
                        # Reconstruct momentum from factored form
                        _unnmf((state['mu_row'], state['mu_col']), out=mt_buf)
                        unpacked_sign = _unpack_bools(state['sign_buf'], original_m=d2)
                        torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)
                        
                        # Momentum update in-place
                        mt_buf.lerp_(grad.view(d1, d2), 1 - momentum)
                        
                        # Orthogonalization (modifies mt_buf in-place)
                        ortho_update = _zeropower_via_newtonschulz(
                            mt_buf, ns_coefficients, ns_steps, eps, use_cans, cans_a_bound
                        )
                        
                        scale_factor = math.sqrt(max(d1, d2)) * rms_scale
                        ortho_update.mul_(scale_factor)
                        
                        # Save state
                        state['sign_buf'] = _pack_bools(mt_buf > 0)
                        _nnmf(mt_buf.abs(), out=(state['mu_row'], state['mu_col']))
                        
                        ortho_update = ortho_update.view(original_shape)
                    else:
                        momentum_buffer = state['momentum_buffer']
                        momentum_buffer.lerp_(grad, 1 - momentum)
                        
                        update_2d = momentum_buffer.view(original_shape[0], -1)
                        ortho_update = _zeropower_via_newtonschulz(
                            update_2d, ns_coefficients, ns_steps, eps, use_cans, cans_a_bound
                        )
                        
                        scale_factor = math.sqrt(max(update_2d.shape[0], update_2d.shape[1])) * rms_scale
                        ortho_update.mul_(scale_factor)
                        ortho_update = ortho_update.view(original_shape)
                    
                    # Apply weight decay (before parameter update)
                    if weight_decay > 0:
                        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                            add_stochastic_(p.data, p.data, alpha=-lr * weight_decay)
                        else:
                            p.add_(p, alpha=-lr * weight_decay)
                    
                    # Apply parameter update
                    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                        add_stochastic_(p.data, ortho_update, alpha=-lr)
                    else:
                        p.add_(ortho_update, alpha=-lr)
                    
                # 1D parameters or parameters without orthogonalization
                else:
                    if state.get('factored', False):
                        d1, d2 = state['effective_shape']
                        mt_buf = state['mt_buf']
                        
                        _unnmf((state['mu_row'], state['mu_col']), out=mt_buf)
                        unpacked_sign = _unpack_bools(state['sign_buf'], original_m=d2)
                        torch.where(unpacked_sign, mt_buf, -mt_buf, out=mt_buf)
                        
                        mt_buf.lerp_(grad.view(d1, d2), 1 - momentum)
                        update = mt_buf.view(grad.shape)
                        
                        state['sign_buf'] = _pack_bools(mt_buf > 0)
                        _nnmf(mt_buf.abs(), out=(state['mu_row'], state['mu_col']))
                    else:
                        update = state['momentum_buffer']
                        update.lerp_(grad, 1 - momentum)
                    
                    # Apply weight decay (before parameter update)
                    if weight_decay > 0:
                        if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                            add_stochastic_(p.data, p.data, alpha=-lr * weight_decay)
                        else:
                            p.add_(p, alpha=-lr * weight_decay)
                    
                    # Apply parameter update
                    if p.dtype == torch.bfloat16 and self.stochastic_rounding:
                        add_stochastic_(p.data, update, alpha=-lr)
                    else:
                        p.add_(update, alpha=-lr)
        
        # Apply QK-Clip (if provided)
        if attention_max_logits is not None:
            self._apply_qk_clip(attention_max_logits)
        
        return loss
    
    @torch.no_grad()
    def _apply_qk_clip(self, attention_max_logits: Dict[str, torch.Tensor]):
        """
        Apply QK-Clip to attention weights based on max logits per head.
        
        Args:
            attention_max_logits: Dict mapping parameter names to max logit per head
                Example structure for tracking:
                {
                    "layer.0.attn.Wq_c": tensor([max_logit_head_0, max_logit_head_1, ...]),
                    "layer.0.attn.Wk_c": tensor([...]),
                }
        """
        tau = self.defaults['qk_clip_threshold']
        alpha = self.defaults['qk_clip_alpha']
        
        for group in self.param_groups:
            for p in group['params']:
                param_name = self._get_param_name(p)
                
                if param_name not in attention_max_logits:
                    continue
                
                max_logits = attention_max_logits[param_name]
                
                # Compute per-head scaling factors: γ_h = min(1, τ / S^h_max) (in-place)
                gamma = (tau / max_logits).clamp_(max=1.0)
                needs_clip = gamma < 1.0
                
                if not needs_clip.any():
                    continue
                
                # Apply different scaling based on component type
                if 'Wq_c' in param_name or 'Wk_c' in param_name:
                    # Head-specific components: scale by sqrt(γ) (in-place)
                    sqrt_gamma = gamma.sqrt()
                    self._scale_attention_heads(p, sqrt_gamma, needs_clip)
                    
                elif 'Wq_r' in param_name:
                    # Query rotary: scale by γ
                    self._scale_attention_heads(p, gamma, needs_clip)
                    
                elif 'Wk_r' in param_name:
                    # Key rotary (shared): don't touch to avoid cross-head effects
                    pass
    
    @staticmethod
    def _scale_attention_heads(
        param: torch.Tensor, 
        scale_factors: torch.Tensor, 
        mask: torch.Tensor
    ):
        """
        Scale specific attention heads in a parameter tensor.
        
        Args:
            param: Parameter tensor (shape depends on architecture)
            scale_factors: Scaling factor per head
            mask: Boolean mask indicating which heads to scale
        """
        # Assumes first dimension is the head dimension
        # This may need adjustment based on actual weight layout
        num_heads = scale_factors.shape[0]
        
        if param.dim() == 2:
            # Shape: [num_heads * head_dim, in_features] or similar
            head_dim = param.shape[0] // num_heads
            
            for h in range(num_heads):
                if mask[h]:
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    param[start_idx:end_idx] *= scale_factors[h]
        
        elif param.dim() == 3:
            # Shape: [num_heads, head_dim, in_features]
            for h in range(num_heads):
                if mask[h]:
                    param[h] *= scale_factors[h]
    
    def _get_param_name(self, param: torch.Tensor) -> str:
        """Helper to get parameter name for tracking."""
        # This would need to be implemented with proper parameter name tracking
        # In practice, you'd maintain a mapping in __init__ or use named_parameters
        if not hasattr(self, '_param_to_name'):
            self._param_to_name = {}
        return self._param_to_name.get(id(param), "")
    
    def register_attention_params(
        self, 
        param_name_mapping: Dict[str, torch.nn.Parameter]
    ):
        """
        Register attention parameters for QK-Clip tracking.
        
        Args:
            param_name_mapping: Dict mapping descriptive names to parameters
                Example: {
                    "layer.0.attn.Wq_c": model.layer[0].attn.Wq_c,
                    "layer.0.attn.Wk_c": model.layer[0].attn.Wk_c,
                }
        """
        if not hasattr(self, '_param_to_name'):
            self._param_to_name = {}
        
        for name, param in param_name_mapping.items():
            self._param_to_name[id(param)] = name