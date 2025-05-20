"""A set of reward utilities written by the authors of dm_control."""

import numpy as np
import torch


def gripper_caging_reward(
    left_pad_y,
    right_pad_y,
    action,
    tcp,
    init_tcp,
    obj_pos,
    obj_init_pos,
    obj_radius,
    pad_success_thresh,
    object_reach_radius,
    xz_thresh,
    desired_gripper_effort=1.0,
    high_density=False,
    medium_density=False,
):
    if high_density and medium_density:
        msg = "Can only be either high_density or medium_density"
        raise ValueError(msg)
    # get current positions of left and right pads (Y axis)
    pad_y_lr = torch.stack((left_pad_y, right_pad_y), dim=-1)
    # compare *current* pad positions with *current* obj position (Y axis)
    pad_to_obj_lr = torch.abs(pad_y_lr - obj_pos[:, 1:2])
    # compare *current* pad positions with *initial* obj position (Y axis)
    pad_to_objinit_lr = torch.abs(pad_y_lr - obj_init_pos[:, 1:2])

    caging_lr_margin = torch.abs(pad_to_objinit_lr - pad_success_thresh)

    caging_lr = [
        tolerance(
            pad_to_obj_lr[:, i],  # "x" in the description above
            bounds=(obj_radius, pad_success_thresh),
            margin=caging_lr_margin[:, i],  # "margin" in the description above
            sigmoid="long_tail",
        )
        for i in range(2)
    ]
    caging_y = hamacher_product(*caging_lr)

    xz = [0, 2]

    caging_xz_margin = torch.norm(obj_init_pos[:, xz] - init_tcp[:, xz], dim=-1)
    caging_xz_margin -= xz_thresh
    caging_xz = tolerance(
        torch.norm(tcp[:, xz] - obj_pos[:, xz], dim=-1),  # "x" in the description above
        bounds=(0, xz_thresh),
        margin=caging_xz_margin,  # "margin" in the description above
        sigmoid="long_tail",
    )

    gripper_closed = action[:, -1].clip(0, desired_gripper_effort) / desired_gripper_effort

    caging = hamacher_product(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, gripper_closed, torch.zeros_like(gripper_closed))
    caging_and_gripping = hamacher_product(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2
    if medium_density:
        tcp_to_obj = torch.norm(obj_pos - tcp, dim=-1)
        tcp_to_obj_init = torch.norm(obj_init_pos - init_tcp, dim=-1)
        # Compute reach reward
        # - We subtract `object_reach_radius` from the margin so that the
        #   reward always starts with a value of 0.1
        reach_margin = torch.abs(tcp_to_obj_init - object_reach_radius)
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=reach_margin,
            sigmoid="long_tail",
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2

    return caging_and_gripping


# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            msg = f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
            raise ValueError(msg)
    elif not 0 < value_at_1 < 1:
        msg = f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
        raise ValueError(msg)

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return torch.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / torch.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, (1 + torch.cos(torch.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return torch.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - torch.tanh(x * scale) ** 2

    else:
        msg = f"Unknown sigmoid type {sigmoid!r}."
        raise ValueError(msg)


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    sigmoid="gaussian",
    value_at_margin=_DEFAULT_VALUE_AT_MARGIN,
):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        msg = "Lower bound must be <= upper bound."
        raise ValueError(msg)

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if isinstance(margin, float):
        if margin < 0:
            msg = f"`margin` must be non-negative. Current value: {margin}"
            raise ValueError(msg)
        if margin == 0:
            value = torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))
        else:
            d = torch.where(x < lower, lower - x, x - upper) / margin
            value = torch.where(in_bounds, torch.ones_like(x), _sigmoids(d, value_at_margin, sigmoid))
    elif isinstance(margin, torch.Tensor):
        # assert (margin >= 0).all(), f"`margin` must be non-negative. Current value: {margin}"
        d = torch.where(x < lower, lower - x, x - upper) / margin
        ones = torch.ones_like(margin)
        zeros = torch.zeros_like(margin)
        value = torch.where(
            margin == 0,
            torch.where(in_bounds, ones, zeros),
            torch.where(in_bounds, ones, _sigmoids(d, value_at_margin, sigmoid)),
        )
    return value


def inverse_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="reciprocal"):
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0)
    return 1 - bound


def rect_prism_tolerance(curr, zero, one):
    """Computes a reward if curr is inside a rectangular prism region.

    The 3d points curr and zero specify 2 diagonal corners of a rectangular
    prism that represents the decreasing region.

    one represents the corner of the prism that has a reward of 1.
    zero represents the diagonal opposite corner of the prism that has a reward
        of 0.
    Curr is the point that the prism reward region is being applied for.

    Args:
        curr(np.ndarray): The point whose reward is being assessed.
            shape is (3,).
        zero(np.ndarray): One corner of the rectangular prism, with reward 0.
            shape is (3,)
        one(np.ndarray): The diagonal opposite corner of one, with reward 1.
            shape is (3,)
    """

    def in_range(a, b, c):
        return float(b <= a <= c) if c >= b else float(c <= a <= b)

    in_prism = (
        in_range(curr[0], zero[0], one[0]) and in_range(curr[1], zero[1], one[1]) and in_range(curr[2], zero[2], one[2])
    )
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
        # return 0.01
    else:
        return 1.0


def hamacher_product(a, b):
    """The hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (float): 1st term of hamacher product.
        b (float): 2nd term of hamacher product.

    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        float: The hammacher product of a and b
    """
    # if not ((0.0 <= a).all() and (a <= 1.0).all() and (0.0 <= b).all() and (b <= 1.0).all()):
    #     msg = "a and b must range between 0 and 1"
    #     raise ValueError(msg)

    denominator = a + b - (a * b)
    h_prod = torch.where(denominator > 0, (a * b) / denominator, torch.zeros_like(denominator))

    # assert (0.0 <= h_prod).all() and (h_prod <= 1.0).all()
    return h_prod
