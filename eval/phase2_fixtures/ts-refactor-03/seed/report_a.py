"""Revenue report."""


def _format_currency(amount):
    return f"${amount:,.2f}"


def render(label, amount):
    return f"{label}: {_format_currency(amount)}"
