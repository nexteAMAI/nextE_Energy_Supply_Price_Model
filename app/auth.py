"""app.auth - sign-in gate for the hosted application.

Credentials are never in code, in the repository or in the session log: they are read from
Streamlit secrets (`.streamlit/secrets.toml` locally, the Secrets box on Streamlit Cloud),
section `[auth]` with `username` and `password`, typed by the administrator. When no `[auth]`
section is configured the gate is off (local development) and the sidebar says so.

    [auth]
    username = "..."
    password = "..."

Comparison is constant-time; the entered password is not kept anywhere; sign-out clears the
session flag. This is an access gate for a confidential tool, not an identity system (the
production access model is decided at the production stage - OPEN_ITEMS DEPLOY).
"""

from __future__ import annotations

import hmac

import streamlit as st

FLAG = "esb_auth_ok"
USER = "esb_auth_user"


def configured() -> tuple[str, str] | None:
    try:
        sec = st.secrets["auth"]
        u, p = str(sec["username"]), str(sec["password"])
        return (u, p) if u and p else None
    except (KeyError, FileNotFoundError, AttributeError):
        return None


def is_signed_in() -> bool:
    return bool(st.session_state.get(FLAG, False))


def sign_out() -> None:
    st.session_state[FLAG] = False
    st.session_state.pop(USER, None)


def gate() -> bool:
    """Render the sign-in screen when required; return True when the app may render."""
    creds = configured()
    if creds is None or is_signed_in():
        return True
    st.markdown('<div class="esb-mark">CONFIDENTIAL · nextE</div>', unsafe_allow_html=True)
    st.markdown("# nextE Energy Supply Bid Management Tool")
    st.markdown('<div class="esb-caption">Sign in to continue. Access is limited to authorised nextE staff.</div>', unsafe_allow_html=True)
    with st.form("esb_sign_in"):
        u = st.text_input("Username", autocomplete="username")
        p = st.text_input("Password", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        good = hmac.compare_digest(u.encode("utf-8"), creds[0].encode("utf-8")) & hmac.compare_digest(p.encode("utf-8"), creds[1].encode("utf-8"))
        if good:
            st.session_state[FLAG] = True
            st.session_state[USER] = u
            st.rerun()
        st.markdown('<div class="esb-refusal">The username or password is not recognised.</div>', unsafe_allow_html=True)
    return False
