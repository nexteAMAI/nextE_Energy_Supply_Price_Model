"""
Authentication module for Layer 3 Streamlit dashboard.

Uses Streamlit's built-in secrets management for password-based access control.
Passwords are configured in Streamlit Cloud → App Settings → Secrets,
or locally in .streamlit/secrets.toml.

The authentication state persists across page navigation via st.session_state.
"""

import streamlit as st
import hmac


def check_password() -> bool:
    """
    Gate the app behind a password form.

    Returns True if the user has entered a valid password.
    Passwords are defined in st.secrets["passwords"] as {username: password} pairs.
    """

    # Already authenticated this session
    if st.session_state.get("authenticated", False):
        return True

    # Check if secrets are configured
    if "passwords" not in st.secrets:
        # No passwords configured — allow open access but show warning
        st.sidebar.warning("⚠️ No access control configured. Set passwords in Streamlit Secrets.")
        return True

    def _on_submit():
        """Validate credentials on form submission."""
        username = st.session_state.get("_auth_user", "")
        password = st.session_state.get("_auth_pass", "")

        passwords = st.secrets["passwords"]

        if username in passwords and hmac.compare_digest(
            password, passwords[username]
        ):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            # Clear password from state
            del st.session_state["_auth_pass"]
        else:
            st.session_state["authenticated"] = False
            st.session_state["_auth_failed"] = True

    # Render login form
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-family: 'DM Sans', sans-serif; color: #1F4E79;">⚡ RO Energy Pricing Model</h1>
        <p style="color: #666; font-family: 'DM Sans', sans-serif;">Restricted access — authorized personnel only</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.text_input("Username", key="_auth_user")
            st.text_input("Password", type="password", key="_auth_pass")
            st.form_submit_button("Sign In", on_click=_on_submit, use_container_width=True)

        if st.session_state.get("_auth_failed", False):
            st.error("Invalid username or password.")

    return False


def show_user_info():
    """Display current user in the sidebar."""
    if st.session_state.get("authenticated"):
        username = st.session_state.get("username", "unknown")
        st.sidebar.markdown(f"**Signed in as:** `{username}`")
        if st.sidebar.button("Sign Out", key="signout"):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()


def require_auth():
    """
    Call at the top of app.py to enforce authentication.
    Returns True if authenticated, otherwise renders login and stops.
    """
    if not check_password():
        st.stop()
    show_user_info()
    return True
