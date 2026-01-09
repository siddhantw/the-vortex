"""
Login and Authentication UI Components
"""

import streamlit as st
from .auth_manager import StreamlitAuthManager
from .rbac_config import COMPLIANCE_CONFIG


def render_login_page(st_auth: StreamlitAuthManager):
    """Render the login page"""

    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .login-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #EC5328, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .login-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .login-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .security-notice {
        background: linear-gradient(135deg, rgba(236, 83, 40, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border-left: 4px solid #EC5328;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)

    # Login container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="login-header">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🌀</div>
        <div class="login-title">The Vortex</div>
        <div class="login-subtitle">Virtual Orchestrator for Real-world Technology Excellence</div>
    </div>
    """, unsafe_allow_html=True)

    # Login box
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    st.markdown("### 🔐 Sign In to Your Account")

    # Login form
    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        remember_me = st.checkbox("Remember me for 7 days", key="remember_me")

        col1, col2 = st.columns([2, 1])

        with col1:
            submit = st.form_submit_button("🚀 Sign In", use_container_width=True)

        with col2:
            help_btn = st.form_submit_button("❓ Help", use_container_width=True)

        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                # Show processing indicator
                with st.spinner("Authenticating..."):
                    success = st_auth.login(username, password)

                if success:
                    user = st_auth.get_current_user()

                    # Check if password change required
                    if user.must_change_password:
                        st.session_state.force_password_change = True
                        st.warning("⚠️ You must change your password before continuing")
                        st.rerun()  # Rerun to show password change form
                    else:
                        # Clear any stale form state completely
                        for key in list(st.session_state.keys()):
                            if key.startswith('login_') or key == 'show_login_help':
                                del st.session_state[key]

                        # Clear navigation states to ensure clean landing on main page
                        st.session_state.show_admin_panel = False
                        st.session_state.show_user_profile = False
                        st.session_state.user_menu_previous = "🧭 Menu"

                        # Set success flags
                        st.session_state.just_logged_in = True
                        st.session_state.login_user_name = user.full_name or user.username

                        # Show immediate feedback
                        st.success(f"✅ Welcome back, {user.full_name or user.username}!")

                        # Trigger immediate rerun without delay
                        st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please try again.")

        if help_btn:
            st.session_state.show_login_help = True

    st.markdown('</div>', unsafe_allow_html=True)  # Close login-box

    # Security notice
    policy = COMPLIANCE_CONFIG['session_policy']
    st.markdown(f"""
    <div class="security-notice">
        <strong>🔒 Security Notice:</strong><br>
        • Sessions expire after {policy['max_session_duration_hours']} hours of activity<br>
        • Auto-logout after {policy['idle_timeout_minutes']} minutes of inactivity<br>
        • Account locked after {policy['max_failed_login_attempts']} failed login attempts<br>
        • All access is logged and monitored for security compliance
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close login-container

    # Help dialog
    if st.session_state.get('show_login_help', False):
        with st.expander("❓ Login Help & Information", expanded=True):
            st.markdown("""
            ### 🆘 Need Help Logging In?
            
            **Default Admin Credentials:**
            - **Username:** `admin`
            - **Password:** `*****************`
            - ⚠️ You will be required to change this password on first login
            
            **Forgot Your Password?**
            Please contact your system administrator to reset your password.
            
            **Account Locked?**
            If you've exceeded the maximum failed login attempts, your account will be locked for 30 minutes.
            Contact your administrator for immediate assistance.
            
            **New User?**
            Please request access from your system administrator who can create an account for you.
            
            **Technical Support:**
            For technical issues, contact: siddhant.wadhwani@newfold.com
            """)

            if st.button("Close Help"):
                st.session_state.show_login_help = False
                st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; color: #94a3b8; font-size: 0.9rem;">
        🌀 The Vortex - Powered by Advanced AI & Automation<br>
        © 2025-26 Newfold Digital. All rights reserved.
    </div>
    """, unsafe_allow_html=True)


def render_password_change_form(st_auth: StreamlitAuthManager):
    """Render forced password change form"""

    st.markdown("""
    <div style="max-width: 600px; margin: 0 auto; padding: 2rem;">
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔐</div>
            <h1>Change Your Password</h1>
            <p style="color: #64748b;">For security reasons, you must change your password</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    user = st_auth.get_current_user()
    user_manager = st_auth.auth_manager.user_manager

    with st.form("password_change_form"):
        st.markdown("### Set New Password")

        old_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")

        # Password requirements
        policy = COMPLIANCE_CONFIG['password_policy']
        st.markdown(f"""
        **Password Requirements:**
        - Minimum {policy['min_length']} characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one number
        - At least one special character (!@#$%^&*(),.?":{{}}|<>)
        - Cannot reuse last {policy['prevent_reuse_count']} passwords
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            submit = st.form_submit_button("✅ Change Password", use_container_width=True)

        with col2:
            logout = st.form_submit_button("🚪 Logout", use_container_width=True)

        if logout:
            st_auth.logout()
            st.rerun()

        if submit:
            if not old_password or not new_password or not confirm_password:
                st.error("Please fill all fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            else:
                try:
                    success = user_manager.change_password(
                        user.username,
                        old_password,
                        new_password
                    )

                    if success:
                        # Clear the force password change flag
                        st.session_state.force_password_change = False

                        # Set success flags for welcome message
                        st.session_state.just_logged_in = True
                        st.session_state.login_user_name = user.full_name or user.username

                        # Show success message
                        st.success("✅ Password changed successfully!")
                        st.balloons()

                        # Clean rerun to main page
                        st.rerun()
                    else:
                        st.error("❌ Current password is incorrect")
                except ValueError as e:
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error changing password: {str(e)}")


def render_user_profile(st_auth: StreamlitAuthManager):
    """Render user profile page"""

    st.title("👤 My Profile")

    user = st_auth.get_current_user()
    if not user:
        st.error("User not found")
        return

    # Profile information
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%); 
                    border-radius: 16px; color: white;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">👤</div>
            <h2 style="color: white; margin: 0;">{}</h2>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem;">@{}</p>
        </div>
        """.format(user.full_name or user.username, user.username), unsafe_allow_html=True)

    with col2:
        st.markdown("### 📋 Profile Information")

        st.markdown(f"**Email:** {user.email}")
        st.markdown(f"**Department:** {user.department or 'Not specified'}")
        st.markdown(f"**Roles:** {', '.join(user.roles)}")
        st.markdown(f"**Groups:** {', '.join(user.groups) if user.groups else 'None'}")
        st.markdown(f"**Account Status:** {'🟢 Active' if user.is_active else '🔴 Inactive'}")
        st.markdown(f"**MFA Enabled:** {'✅ Yes' if user.mfa_enabled else '❌ No'}")

    st.divider()

    # Activity statistics
    st.markdown("### 📊 Activity Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Executions", user.total_executions)

    with col2:
        st.metric("Today's Executions", user.daily_execution_count)

    with col3:
        max_daily = user.get_max_daily_executions()
        remaining = max(0, max_daily - user.daily_execution_count)
        st.metric("Remaining Today", remaining)

    with col4:
        st.metric("Active Sessions", len(user.active_sessions))

    st.divider()

    # Account management
    st.markdown("### ⚙️ Account Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔑 Change Password", use_container_width=True):
            st.session_state.show_profile_password_change = True

    with col2:
        if st.button("🔒 Enable MFA", use_container_width=True, disabled=user.mfa_enabled):
            st.info("MFA setup will be available in the next update")

    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st_auth.logout()
            st.rerun()

    # Change password form
    if st.session_state.get('show_profile_password_change', False):
        st.divider()
        with st.expander("🔑 Change Password", expanded=True):
            with st.form("profile_password_change"):
                old_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")

                col1, col2 = st.columns(2)

                with col1:
                    if st.form_submit_button("✅ Change", use_container_width=True):
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        else:
                            try:
                                user_manager = st_auth.auth_manager.user_manager
                                success = user_manager.change_password(
                                    user.username,
                                    old_password,
                                    new_password
                                )
                                if success:
                                    st.success("Password changed successfully!")
                                    st.session_state.show_profile_password_change = False
                                else:
                                    st.error("Current password is incorrect")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                with col2:
                    if st.form_submit_button("❌ Cancel", use_container_width=True):
                        st.session_state.show_profile_password_change = False
                        st.rerun()

    st.divider()

    # Recent activity
    st.markdown("### 📜 Recent Activity")

    audit_logger = st_auth.auth_manager.audit_logger
    recent_events = audit_logger.get_user_activity(user.username, days=7)

    if recent_events:
        import pandas as pd
        events_data = []
        for event in recent_events[:20]:  # Show last 20
            events_data.append({
                'Timestamp': event.timestamp,
                'Action': event.action,
                'Module': event.module_id or 'N/A',
                'Status': '✅' if event.success else '❌'
            })

        df = pd.DataFrame(events_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent activity")

