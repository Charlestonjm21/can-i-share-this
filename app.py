"""Streamlit front end for Can I Share This?

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from cish import agent, directory, limits, llm
from cish.schema import Verdict

VERDICT_STYLE = {
    Verdict.SAFE: ("#1b5e20", "#e8f5e9", "SAFE"),
    Verdict.CHECK_POLICY: ("#7a4f01", "#fff4e0", "CHECK POLICY"),
    Verdict.DO_NOT_SHARE: ("#8e1414", "#fdecec", "DO NOT SHARE"),
}

st.set_page_config(page_title="Can I Share This?", page_icon="🔐")


@st.cache_resource
def daily_counter() -> limits.DailyCounter:
    """One counter shared by every visitor of this running app."""
    return limits.DailyCounter()


st.title("Can I Share This?")
st.caption(
    "Advisory only. This tool evaluates a described action against company "
    "policy. It never receives, stores, or transmits the data itself."
)

with st.sidebar:
    st.subheader("Signed in as")
    people = directory.all_employees()
    labels = [f"{p['name']} — {p['role']}" for p in people]
    choice = st.selectbox("Employee", labels, index=0)
    requester = people[labels.index(choice)]
    st.caption(f"ID {requester['id']} · role: {requester['role']}")
    st.divider()
    st.subheader("Model")
    st.caption(f"Provider: `{llm.provider()}`")
    st.caption(f"Model: `{llm.active_model()}`")
    if llm.available():
        st.success("Provider reachable", icon="✅")
    else:
        st.warning(
            f"No credentials found for `{llm.provider()}`. Set the API key "
            f"in your environment, or switch CISH_PROVIDER.",
            icon="⚠️",
        )
    st.divider()
    st.caption(
        "Authorization is read from the directory, not from what you type. "
        "Switch users to see how the same question changes answer."
    )

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("e.g. Can I upload this customer list to ChatGPT?")

if question:
    cap = limits.max_questions()
    asked = st.session_state.get("asked", 0)
    if len(question) > limits.max_chars():
        st.warning(f"Please keep the question under {limits.max_chars()} "
                   f"characters. Describe the action, do not paste the data.")
        st.stop()
    if cap and asked >= cap:
        st.warning(f"This demo allows {cap} questions per session. "
                   f"Refresh the page to start a new session.")
        st.stop()
    if not daily_counter().try_take(limits.daily_cap()):
        st.warning("The demo has reached its daily limit to protect a shared "
                   "free quota. Please try again tomorrow, or run it "
                   "locally with your own key (see the README).")
        st.stop()
    st.session_state.asked = asked + 1
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking policy…"):
            try:
                decision = agent.decide(question, requester["id"])
            except Exception as exc:
                st.error(f"Could not evaluate: {exc}")
                st.stop()

        if decision.clarifying_question:
            st.markdown(decision.clarifying_question)
            st.session_state.history.append(
                {"role": "assistant", "content": decision.clarifying_question}
            )
        else:
            fg, bg, label = VERDICT_STYLE[decision.verdict]
            st.markdown(
                f"<div style='background:{bg};color:{fg};padding:10px 14px;"
                f"border-radius:6px;font-weight:700;display:inline-block'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(decision.rationale)

            if decision.provisions:
                with st.expander("Policy clauses this is based on"):
                    for p in decision.provisions:
                        st.markdown(
                            f"**{p.clause_id} — {p.heading}**  \n"
                            f"*{p.doc_title}*  \n{p.text}"
                        )
            else:
                st.info(
                    "No clause in the policy corpus covers this action. "
                    "That gap is itself worth reporting to the policy owner."
                )

            with st.expander("How this was evaluated"):
                s = decision.slots
                st.markdown(
                    f"- Data type: `{s.data_type.value}`\n"
                    f"- Recipient: `{s.recipient.value}`\n"
                    f"- Channel: `{s.channel.value}`\n"
                    f"- Volume: `{s.volume.value}`\n"
                    f"- Rule engine said: `{decision.rule_verdict.value}`\n"
                    f"- Model said: `{decision.model_verdict.value if decision.model_verdict else 'not consulted (a hard rule already blocked this)'}`\n"
                    f"- Final verdict is the more cautious of the two.\n"
                    f"- Model used: `{llm.active_model()}`"
                )

            st.session_state.history.append(
                {"role": "assistant",
                 "content": f"**{label}** — {decision.rationale}"}
            )
