"""
Admin Panel for User and Role Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from .auth_manager import StreamlitAuthManager
from .rbac_config import SYSTEM_ROLES, Permission
from .audit_logger import AuditAction, AuditSeverity


def render_admin_panel(st_auth: StreamlitAuthManager):
    """Render the admin panel"""

    # Initialize session state for admin panel
    if 'show_create_user' not in st.session_state:
        st.session_state.show_create_user = False
    if 'show_reset_password' not in st.session_state:
        st.session_state.show_reset_password = False
    if 'show_edit_user' not in st.session_state:
        st.session_state.show_edit_user = False
    if 'confirm_delete_user' not in st.session_state:
        st.session_state.confirm_delete_user = None

    # SECURITY: Require admin role (super_admin or admin only)
    # This ensures only users with super_admin or admin roles can access the admin panel
    st_auth.require_admin_access()

    st.title("🔐 Admin Panel - User & Access Management")

    # Admin tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 User Management",
        "🔑 Roles & Permissions",
        "📊 Audit Logs",
        "📈 Usage Analytics",
        "⚙️ Security Settings"
    ])

    with tab1:
        render_user_management(st_auth)

    with tab2:
        render_roles_permissions(st_auth)

    with tab3:
        render_audit_logs(st_auth)

    with tab4:
        render_usage_analytics(st_auth)

    with tab5:
        render_security_settings(st_auth)


def render_user_management(st_auth: StreamlitAuthManager):
    """User management interface"""
    st.header("User Management")

    user_manager = st_auth.auth_manager.user_manager
    audit_logger = st_auth.auth_manager.audit_logger
    current_user = st_auth.get_current_user()

    # Initialize refresh timestamp for auto-refresh
    if 'user_list_refresh_time' not in st.session_state:
        st.session_state.user_list_refresh_time = datetime.now()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Header with last refresh time
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.subheader("👥 All Users")
        with header_col2:
            if 'user_list_refresh_time' in st.session_state:
                time_diff = (datetime.now() - st.session_state.user_list_refresh_time).total_seconds()
                if time_diff < 60:
                    st.caption(f"🔄 Refreshed {int(time_diff)}s ago")
                else:
                    st.caption(f"🔄 Refreshed {int(time_diff/60)}m ago")

        # Fetch users (always fresh data)
        users = user_manager.list_users()

        if users:
            df = pd.DataFrame(users)
            # Select relevant columns
            display_cols = ['username', 'full_name', 'email', 'roles', 'department',
                          'is_active', 'last_login', 'total_executions']
            available_cols = [col for col in display_cols if col in df.columns]

            # Use timestamp in key to force refresh when data changes
            st.dataframe(
                df[available_cols],
                use_container_width=True,
                key=f"user_list_{st.session_state.user_list_refresh_time.timestamp()}"
            )
        else:
            st.info("No users found")

    with col2:
        st.subheader("➕ Quick Actions")

        if st.button("🆕 Create New User", use_container_width=True):
            st.session_state.show_create_user = True

        if st.button("🔄 Refresh List", use_container_width=True):
            # Update refresh timestamp for cache busting
            st.session_state.user_list_refresh_time = datetime.now()
            st.rerun()

        # Statistics
        st.metric("Total Users", len(users))
        active_users = sum(1 for u in users if u.get('is_active', False))
        st.metric("Active Users", active_users)

    # Create user form
    if st.session_state.get('show_create_user', False):
        st.divider()
        with st.expander("🆕 Create New User", expanded=True):
            with st.form("create_user_form"):
                col1, col2 = st.columns(2)

                with col1:
                    new_username = st.text_input("Username*", key="new_username")
                    new_email = st.text_input("Email*", key="new_email")
                    new_password = st.text_input("Password*", type="password", key="new_password")
                    new_confirm_password = st.text_input("Confirm Password*", type="password", key="new_confirm_password")

                with col2:
                    new_full_name = st.text_input("Full Name", key="new_full_name")
                    new_department = st.text_input("Department", key="new_department")
                    new_roles = st.multiselect(
                        "Roles*",
                        options=list(SYSTEM_ROLES.keys()),
                        key="new_roles"
                    )
                    new_groups = st.text_input("Groups (comma-separated)", key="new_groups")

                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    submit = st.form_submit_button("✅ Create User", use_container_width=True)

                with col2:
                    cancel = st.form_submit_button("❌ Cancel", use_container_width=True)

                if cancel:
                    st.session_state.show_create_user = False
                    st.rerun()

                if submit:
                    if new_password != new_confirm_password:
                        st.error("Passwords do not match!")
                    elif not new_username or not new_email or not new_password or not new_roles:
                        st.error("Please fill all required fields marked with *")
                    else:
                        try:
                            groups_list = [g.strip() for g in new_groups.split(',')] if new_groups else []

                            user_manager.create_user(
                                username=new_username,
                                email=new_email,
                                password=new_password,
                                roles=new_roles,
                                full_name=new_full_name,
                                department=new_department,
                                groups=groups_list,
                                created_by=current_user.username
                            )

                            # Log action
                            audit_logger.log(
                                action=AuditAction.USER_CREATE,
                                username=current_user.username,
                                user_id=current_user.user_id,
                                success=True,
                                severity=AuditSeverity.INFO,
                                affected_user=new_username,
                                details={
                                    'roles': new_roles,
                                    'department': new_department
                                },
                                session_id=st.session_state.auth_session_id
                            )

                            st.success(f"✅ User '{new_username}' created successfully!")
                            st.session_state.show_create_user = False
                            # Update refresh timestamp to trigger auto-refresh
                            st.session_state.user_list_refresh_time = datetime.now()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error creating user: {str(e)}")

    # User details and edit
    st.divider()
    st.subheader("🔍 User Details & Management")

    selected_user = st.selectbox(
        "Select user to manage",
        options=[u['username'] for u in users],
        key="selected_user_manage"
    )

    if selected_user:
        user = user_manager.get_user(selected_user)
        if user:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**Username:** {user.username}")
                st.markdown(f"**Email:** {user.email}")
                st.markdown(f"**Full Name:** {user.full_name or 'N/A'}")

            with col2:
                st.markdown(f"**Department:** {user.department or 'N/A'}")
                st.markdown(f"**Roles:** {', '.join(user.roles)}")
                st.markdown(f"**Status:** {'🟢 Active' if user.is_active else '🔴 Inactive'}")

            with col3:
                st.markdown(f"**Last Login:** {user.last_login or 'Never'}")
                st.markdown(f"**Total Executions:** {user.total_executions}")
                st.markdown(f"**Created:** {user.created_at[:10]}")

            # Actions
            st.markdown("### Actions")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if user.is_active:
                    if st.button("🚫 Deactivate", key="deactivate_user"):
                        user_manager.update_user(selected_user, is_active=False)
                        audit_logger.log(
                            action=AuditAction.USER_UPDATE,
                            username=current_user.username,
                            user_id=current_user.user_id,
                            success=True,
                            severity=AuditSeverity.WARNING,
                            affected_user=selected_user,
                            changes={'is_active': False},
                            session_id=st.session_state.auth_session_id
                        )
                        st.success(f"User '{selected_user}' deactivated")
                        # Update refresh timestamp to trigger auto-refresh
                        st.session_state.user_list_refresh_time = datetime.now()
                        st.rerun()
                else:
                    if st.button("✅ Activate", key="activate_user"):
                        user_manager.update_user(selected_user, is_active=True)
                        audit_logger.log(
                            action=AuditAction.USER_UPDATE,
                            username=current_user.username,
                            user_id=current_user.user_id,
                            success=True,
                            severity=AuditSeverity.INFO,
                            affected_user=selected_user,
                            changes={'is_active': True},
                            session_id=st.session_state.auth_session_id
                        )
                        st.success(f"User '{selected_user}' activated")
                        # Update refresh timestamp to trigger auto-refresh
                        st.session_state.user_list_refresh_time = datetime.now()
                        st.rerun()

            with col2:
                if st.button("🔑 Reset Password", key="reset_password"):
                    st.session_state.show_reset_password = True

            with col3:
                if st.button("✏️ Edit User", key="edit_user"):
                    st.session_state.show_edit_user = True

            with col4:
                if st.button("🗑️ Delete User", key="delete_user"):
                    st.session_state.confirm_delete_user = selected_user

            # Password reset form
            if st.session_state.get('show_reset_password', False):
                with st.form("reset_password_form"):
                    st.markdown(f"### Reset Password for {selected_user}")
                    new_password = st.text_input("New Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("Reset"):
                            if new_password != confirm_password:
                                st.error("Passwords do not match!")
                            else:
                                try:
                                    user_manager.reset_password(
                                        selected_user,
                                        new_password,
                                        current_user.username
                                    )
                                    audit_logger.log(
                                        action=AuditAction.PASSWORD_RESET,
                                        username=current_user.username,
                                        user_id=current_user.user_id,
                                        success=True,
                                        severity=AuditSeverity.WARNING,
                                        affected_user=selected_user,
                                        session_id=st.session_state.auth_session_id
                                    )
                                    st.success("Password reset successfully!")
                                    st.session_state.show_reset_password = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    with col2:
                        if st.form_submit_button("Cancel"):
                            st.session_state.show_reset_password = False
                            st.rerun()

            # Delete confirmation
            if st.session_state.get('confirm_delete_user') == selected_user:
                st.warning(f"⚠️ Are you sure you want to delete user '{selected_user}'? This action cannot be undone!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Delete", key="confirm_delete"):
                        user_manager.delete_user(selected_user)
                        audit_logger.log(
                            action=AuditAction.USER_DELETE,
                            username=current_user.username,
                            user_id=current_user.user_id,
                            success=True,
                            severity=AuditSeverity.ERROR,
                            affected_user=selected_user,
                            session_id=st.session_state.auth_session_id
                        )
                        st.success(f"User '{selected_user}' deleted")
                        st.session_state.confirm_delete_user = None
                        # Update refresh timestamp to trigger auto-refresh
                        st.session_state.user_list_refresh_time = datetime.now()
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", key="cancel_delete"):
                        st.session_state.confirm_delete_user = None
                        st.rerun()


def render_roles_permissions(st_auth: StreamlitAuthManager):
    """Roles and permissions interface"""
    st.header("Roles & Permissions")

    st.markdown("""
    This section displays the system roles and their associated permissions.
    Role management and custom role creation will be added in future updates.
    """)

    # Display all roles
    for role_name, role in SYSTEM_ROLES.items():
        with st.expander(f"{role.display_name} ({role_name})", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Description:** {role.description}")
                st.markdown(f"**Priority Level:** {'⭐' * role.priority_level}")
                st.markdown(f"**Max Daily Executions:** {role.max_daily_executions}")
                st.markdown(f"**Max Concurrent Sessions:** {role.max_concurrent_sessions}")
                st.markdown(f"**AI Features:** {'✅ Enabled' if role.can_use_ai_features else '❌ Disabled'}")
                st.markdown(f"**System Role:** {'🔒 Yes' if role.is_system_role else '📝 Custom'}")

            with col2:
                st.markdown("**Permissions:**")
                for perm in sorted(role.permissions, key=lambda p: p.value):
                    st.markdown(f"- ✅ {perm.value}")

                st.markdown("**Module Access:**")
                accessible = role.get_accessible_modules()
                if "*" in accessible:
                    st.markdown("- 🌐 **All Modules**")
                else:
                    for module in sorted(accessible):
                        st.markdown(f"- 📦 {module}")


def render_audit_logs(st_auth: StreamlitAuthManager):
    """Audit logs interface"""
    st.header("Audit Logs")

    st_auth.require_permission(Permission.VIEW_AUDIT_LOGS)

    audit_logger = st_auth.auth_manager.audit_logger

    col1, col2, col3 = st.columns(3)

    with col1:
        days = st.selectbox("Time Range", [1, 7, 30, 90], index=1)

    with col2:
        action_filter = st.selectbox(
            "Action Type",
            ["All"] + [action.value for action in AuditAction]
        )

    with col3:
        user_filter = st.text_input("Username Filter")

    # Get events
    start_date = datetime.now() - timedelta(days=days)
    events = audit_logger.get_events(
        start_date=start_date,
        username=user_filter if user_filter else None,
        limit=500
    )

    # Filter by action if specified
    if action_filter != "All":
        events = [e for e in events if e.action == action_filter]

    st.markdown(f"### 📋 Showing {len(events)} events")

    if events:
        # Convert to DataFrame
        events_data = []
        for event in events:
            events_data.append({
                'Timestamp': event.timestamp,
                'Username': event.username,
                'Action': event.action,
                'Severity': event.severity,
                'Module': event.module_id or 'N/A',
                'Success': '✅' if event.success else '❌',
                'Details': str(event.details)[:50] + '...' if event.details else ''
            })

        df = pd.DataFrame(events_data)
        st.dataframe(df, use_container_width=True)

        # Export option
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"audit_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No audit events found for the selected filters")


def render_usage_analytics(st_auth: StreamlitAuthManager):
    """Usage analytics interface"""
    st.header("Usage Analytics")

    audit_logger = st_auth.auth_manager.audit_logger
    user_manager = st_auth.auth_manager.user_manager

    days = st.selectbox("Analysis Period", [7, 30, 90], index=0, key="analytics_days")

    stats = audit_logger.get_statistics(days=days)

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Events", stats['total_events'])

    with col2:
        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")

    with col3:
        st.metric("Failed Events", stats['failed_events'])

    with col4:
        st.metric("Security Events", stats['security_events'])

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Events by Action")
        if stats['by_action']:
            df = pd.DataFrame(list(stats['by_action'].items()), columns=['Action', 'Count'])
            df = df.sort_values('Count', ascending=False)
            st.bar_chart(df.set_index('Action'))

    with col2:
        st.markdown("### 👥 Events by User")
        if stats['by_user']:
            df = pd.DataFrame(list(stats['by_user'].items()), columns=['User', 'Count'])
            df = df.sort_values('Count', ascending=False).head(10)
            st.bar_chart(df.set_index('User'))


def render_security_settings(st_auth: StreamlitAuthManager):
    """Security settings interface"""
    st.header("Security Settings")

    st_auth.require_permission(Permission.MANAGE_SETTINGS)

    st.markdown("""
    ### 🔐 Current Security Configuration
    
    These settings are configured in the RBAC system and control security policies.
    """)

    from .rbac_config import COMPLIANCE_CONFIG

    # Password Policy
    with st.expander("🔑 Password Policy", expanded=True):
        policy = COMPLIANCE_CONFIG['password_policy']
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Minimum Length:** {policy['min_length']} characters")
            st.markdown(f"**Require Uppercase:** {'✅' if policy['require_uppercase'] else '❌'}")
            st.markdown(f"**Require Lowercase:** {'✅' if policy['require_lowercase'] else '❌'}")

        with col2:
            st.markdown(f"**Require Numbers:** {'✅' if policy['require_numbers'] else '❌'}")
            st.markdown(f"**Require Special Characters:** {'✅' if policy['require_special_chars'] else '❌'}")
            st.markdown(f"**Password Expiry:** {policy['max_age_days']} days")
            st.markdown(f"**Prevent Reuse:** Last {policy['prevent_reuse_count']} passwords")

    # Session Policy
    with st.expander("🕐 Session Policy", expanded=True):
        policy = COMPLIANCE_CONFIG['session_policy']
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Max Session Duration:** {policy['max_session_duration_hours']} hours")
            st.markdown(f"**Idle Timeout:** {policy['idle_timeout_minutes']} minutes")
            st.markdown(f"**Max Failed Login Attempts:** {policy['max_failed_login_attempts']}")

        with col2:
            st.markdown(f"**Lockout Duration:** {policy['lockout_duration_minutes']} minutes")
            st.markdown(f"**MFA Required for Roles:** {', '.join(policy['require_mfa_for_roles'])}")

    # Audit Policy
    with st.expander("📝 Audit Policy", expanded=True):
        policy = COMPLIANCE_CONFIG['audit_policy']
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Log All Access:** {'✅' if policy['log_all_access'] else '❌'}")
            st.markdown(f"**Log Data Exports:** {'✅' if policy['log_data_exports'] else '❌'}")

        with col2:
            st.markdown(f"**Log Config Changes:** {'✅' if policy['log_config_changes'] else '❌'}")
            st.markdown(f"**Log Failed Access:** {'✅' if policy['log_failed_access'] else '❌'}")
            st.markdown(f"**Retention Period:** {policy['retention_days']} days")

