# pyright: reportUnknownMemberType=false

from stages.visual_two_choice_training.models import ResultEnum

RESULT_ORDER = [
    ResultEnum.CORRECT,
    ResultEnum.INCORRECT,
    ResultEnum.BACKGROUND_TOUCHED,
    ResultEnum.TIMEOUT,
    ResultEnum.CANCELLED,
    ResultEnum.PENDING,
]
RESULT_PALETTE = {
    ResultEnum.CORRECT.value: "#16A34A",
    ResultEnum.INCORRECT.value: "#DC2626",
    ResultEnum.BACKGROUND_TOUCHED.value: "#F59E0B",
    ResultEnum.TIMEOUT.value: "#2563EB",
    ResultEnum.CANCELLED.value: "#64748B",
    ResultEnum.PENDING.value: "#A855F7",
}
