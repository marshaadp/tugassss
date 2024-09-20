import streamlit as st

# Initialize session state to track login
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = None

# Set title for the login page
st.title("Sistem Login - Pilih Sisi Akses")

# If not logged in, show the login options
if st.session_state['login_status'] is None:
    # Create two options for the login: Admin and User
    login_option = st.selectbox(
        "Pilih akses masuk:",
        ["Pilih akses", "Admin", "User"]
    )

    # Button for login
    if st.button("Masuk"):
        if login_option == "Admin":
            st.session_state['login_status'] = 'admin'
            st.experimental_set_query_params(status="admin")  # Set query param for reload
        elif login_option == "User":
            st.session_state['login_status'] = 'user'
            st.experimental_set_query_params(status="user")  # Set query param for reload
        else:
            st.warning("Silakan pilih akses terlebih dahulu.")

# Check the status and show the appropriate interface
query_params = st.experimental_get_query_params()

if st.session_state['login_status'] == 'admin' or query_params.get("status") == ["admin"]:
    st.write("Anda masuk sebagai Admin")
    # Redirect to admin app functionality (import or load admin.py)
    import app  # Make sure 'app.py' is in the same directory or add correct path

elif st.session_state['login_status'] == 'user' or query_params.get("status") == ["user"]:
    st.write("Anda masuk sebagai User")
    # Redirect to user app functionality (import or load user.py)
    import user  # Make sure 'user.py' is in the same directory or add correct path

# Option to log out
if st.session_state['login_status'] is not None:
    if st.button("Logout"):
        st.session_state['login_status'] = None
        st.experimental_set_query_params()  # Clear query params for logout
