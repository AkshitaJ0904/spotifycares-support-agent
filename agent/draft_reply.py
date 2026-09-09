"""Grounded reply drafting: conditions the LLM on real historical (customer, support)
pairs retrieved for this message, rather than letting it free-generate from
parametric knowledge. This is the "grounded in how the brand has historically
resolved similar issues" requirement."""
from agent.llm import generate_text

_SYSTEM = (
    "You write public Twitter replies as SpotifyCares, Spotify's customer-support account. "
    "Match the brand voice you see in the EXAMPLES below: warm, casual, brief (1-3 sentences), "
    "starts with a friendly greeting, uses 'Hey there!' / the customer's name-style openers sparingly. "
    "Follow the *pattern* of how similar past issues were actually resolved -- if the examples route "
    "the customer to DM for account-specific help, do the same rather than inventing an on-the-spot fix. "
    "Never invent specific facts (dates, ticket numbers, policy details, refund amounts) that aren't "
    "supported by the examples or the message itself. "
    "The EXAMPLES have been redacted for this exercise: real URLs were replaced with the literal text "
    "'[link]' and agent sign-off codes like '/PK' or '^CG' were stripped. These are artifacts of "
    "redaction, NOT real brand style -- never output the literal string '[link]' or invent a fake "
    "sign-off code; just omit them. Output ONLY the reply text, no preamble."
)


def draft_reply(message: str, intent_name: str, action: str, precedents: list) -> str:
    ex_block = "\n\n".join(
        f"EXAMPLE {i+1} (similarity={p['similarity']:.2f}):\n"
        f"Customer: {p['customer_msg']}\n"
        f"SpotifyCares: {p['support_reply']}"
        for i, p in enumerate(precedents)
    ) or "(no close precedent found)"

    guidance = (
        "This message has been routed for ESCALATION (needs human/DM follow-up for account-specific "
        "verification). Draft the public holding reply that acknowledges the issue and directs them to "
        "next steps (e.g. DM), consistent with the examples -- do not attempt to resolve it fully in public."
        if action == "escalate"
        else "This message will be AUTO-HANDLED. Draft a complete, self-contained helpful reply that "
        "resolves or clearly addresses the issue, consistent with how similar cases were handled."
    )

    prompt = (
        f"Intent: {intent_name}\n\n"
        f"Historical precedent (real past {'→'} resolutions for similar messages):\n{ex_block}\n\n"
        f"New customer message:\n{message!r}\n\n"
        f"{guidance}\n\nWrite the reply now."
    )
    return generate_text(prompt, system=_SYSTEM, temperature=0.5)


if __name__ == "__main__":
    from agent.retrieve import retrieve

    msg = "I was charged twice for premium this month, can you refund me"
    prec = retrieve(msg, k=3)
    print(draft_reply(msg, "Billing & Subscription", "escalate", prec))
