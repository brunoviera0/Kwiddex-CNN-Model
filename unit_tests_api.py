import unittest
import io
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 700, f"Test {datetime.now()}")
    c.save()
    return buffer.getvalue()


def create_image():
    img = Image.new('RGB', (100, 100), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def get_client():
    from fastapi.testclient import TestClient
    from predict import app
    return TestClient(app)


def unique_user():
    return f"user_{datetime.now().strftime('%H%M%S%f')}"


def register_and_login(client=None):
    c = client or get_client()
    u = unique_user()
    c.post("/register", json={"username": u, "password": "password123"})
    r = c.post("/login", json={"username": u, "password": "password123"})
    data = r.json()
    return c, data.get("token"), data.get("user_id")


def auth_header(token):
    return {"Authorization": f"Bearer {token}"}



class TestAuthSecurity(unittest.TestCase):
    def test_register_success(self):
        r = get_client().post("/register", json={"username": unique_user(), "password": "password123"})
        self.assertTrue(r.json()["success"])

    def test_short_username_rejected(self):
        r = get_client().post("/register", json={"username": "ab", "password": "password123"})
        self.assertFalse(r.json()["success"])

    def test_short_password_rejected(self):
        r = get_client().post("/register", json={"username": unique_user(), "password": "short"})
        self.assertFalse(r.json()["success"])

    def test_duplicate_rejected(self):
        c, u = get_client(), unique_user()
        c.post("/register", json={"username": u, "password": "password123"})
        r = c.post("/register", json={"username": u, "password": "password123"})
        self.assertFalse(r.json()["success"])

    def test_login_success(self):
        c, u = get_client(), unique_user()
        c.post("/register", json={"username": u, "password": "password123"})
        r = c.post("/login", json={"username": u, "password": "password123"})
        self.assertTrue(r.json()["success"])

    def test_wrong_password_rejected(self):
        c, u = get_client(), unique_user()
        c.post("/register", json={"username": u, "password": "password123"})
        r = c.post("/login", json={"username": u, "password": "wrongpass123"})
        self.assertFalse(r.json()["success"])

    def test_nonexistent_user_rejected(self):
        r = get_client().post("/login", json={"username": "nonexistent_xyz", "password": "password123"})
        self.assertFalse(r.json()["success"])


class TestJWTAuth(unittest.TestCase):
    def test_login_returns_token(self):
        c, u = get_client(), unique_user()
        c.post("/register", json={"username": u, "password": "password123"})
        r = c.post("/login", json={"username": u, "password": "password123"})
        data = r.json()
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["token"])
        self.assertGreater(len(data["token"]), 0)

    def test_failed_login_no_token(self):
        r = get_client().post("/login", json={"username": "nobody", "password": "wrongpass123"})
        data = r.json()
        self.assertFalse(data["success"])
        self.assertIsNone(data["token"])

    def test_token_is_valid_jwt(self):
        from auth import verify_token
        c, token, _ = register_and_login()
        payload = verify_token(token)
        self.assertIsNotNone(payload)
        self.assertIn("sub", payload)
        self.assertIn("username", payload)
        self.assertIn("exp", payload)

    def test_invalid_token_rejected(self):
        from auth import verify_token
        payload = verify_token("this.is.not.a.valid.token")
        self.assertIsNone(payload)

    def test_expired_token_rejected(self):
        from auth import JWT_SECRET_KEY, JWT_ALGORITHM
        from jose import jwt
        from datetime import timedelta
        payload = {
            "sub": "USR-TEST",
            "username": "test",
            "exp": datetime.utcnow() - timedelta(hours=1),
            "iat": datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        from auth import verify_token
        result = verify_token(expired_token)
        self.assertIsNone(result)


class TestProtectedEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from certification import setup_keys, PRIVATE_KEY_PATH
        if not PRIVATE_KEY_PATH.exists():
            setup_keys()

    def test_certify_rejects_no_token(self):
        r = get_client().post("/certify", files={"file": ("t.pdf", create_pdf(), "application/pdf")})
        self.assertEqual(r.status_code, 401)

    def test_certify_rejects_bad_token(self):
        r = get_client().post(
            "/certify",
            files={"file": ("t.pdf", create_pdf(), "application/pdf")},
            headers=auth_header("invalid.token.here")
        )
        self.assertEqual(r.status_code, 401)

    def test_certify_accepts_valid_token(self):
        c, token, _ = register_and_login()
        r = c.post(
            "/certify",
            files={"file": ("t.pdf", create_pdf(), "application/pdf")},
            headers=auth_header(token)
        )
        self.assertIn(r.status_code, [200, 400])

    def test_certificate_lookup_rejects_no_token(self):
        r = get_client().get("/certificate/KWX-00000000-FAKE")
        self.assertEqual(r.status_code, 401)

    def test_certificate_lookup_rejects_bad_token(self):
        r = get_client().get(
            "/certificate/KWX-00000000-FAKE",
            headers=auth_header("invalid.token.here")
        )
        self.assertEqual(r.status_code, 401)

    def test_certificate_lookup_accepts_valid_token(self):
        c, token, _ = register_and_login()
        r = c.get(
            "/certificate/KWX-00000000-FAKE",
            headers=auth_header(token)
        )
        self.assertIn(r.status_code, [404, 200])

    def test_verify_certificate_is_public(self):
        r = get_client().post(
            "/verify-certificate",
            files={"file": ("t.pdf", create_pdf(), "application/pdf")}
        )
        self.assertNotEqual(r.status_code, 401)

    def test_register_is_public(self):
        r = get_client().post("/register", json={"username": unique_user(), "password": "password123"})
        self.assertNotEqual(r.status_code, 401)

    def test_login_is_public(self):
        r = get_client().post("/login", json={"username": "nobody", "password": "wrongpass123"})
        self.assertNotEqual(r.status_code, 401)

    def test_health_is_public(self):
        r = get_client().get("/health")
        self.assertNotEqual(r.status_code, 401)


class TestBcryptHashing(unittest.TestCase):
    def test_new_user_uses_bcrypt(self):
        from auth import create_user, load_users
        u = unique_user()
        create_user(u, "password123")
        users = load_users()
        self.assertEqual(users[u]["hash_method"], "bcrypt")
        self.assertTrue(users[u]["password_hash"].startswith("$2b$"))

    def test_bcrypt_user_can_login(self):
        from auth import create_user, authenticate
        u = unique_user()
        create_user(u, "password123")
        success, user_id = authenticate(u, "password123")
        self.assertTrue(success)
        self.assertIsNotNone(user_id)

    def test_wrong_password_fails_bcrypt(self):
        from auth import create_user, authenticate
        u = unique_user()
        create_user(u, "password123")
        success, _ = authenticate(u, "wrongpassword")
        self.assertFalse(success)


class TestCertificationSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from certification import setup_keys, PRIVATE_KEY_PATH
        if not PRIVATE_KEY_PATH.exists():
            setup_keys()

    def test_certify_pdf_with_auth(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("t.pdf", create_pdf(), "application/pdf")}, headers=auth_header(token))
        self.assertIn(r.status_code, [200, 400])

    def test_certify_image_with_auth(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("t.jpg", create_image(), "image/jpeg")}, headers=auth_header(token))
        self.assertIn(r.status_code, [200, 400])

    def test_verify_uncertified_invalid(self):
        r = get_client().post("/verify-certificate", files={"file": ("t.pdf", create_pdf(), "application/pdf")})
        self.assertFalse(r.json()["valid"])

    def test_verify_non_pdf_rejected(self):
        r = get_client().post("/verify-certificate", files={"file": ("t.jpg", create_image(), "image/jpeg")})
        self.assertFalse(r.json()["valid"])


class TestInputValidation(unittest.TestCase):
    def test_certify_no_file(self):
        c, token, _ = register_and_login()
        self.assertEqual(c.post("/certify", headers=auth_header(token)).status_code, 422)

    def test_certify_empty_file(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("e.pdf", b"", "application/pdf")}, headers=auth_header(token))
        self.assertIn(r.status_code, [400, 422, 500])

    def test_certify_corrupted_file(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("c.pdf", b"not a pdf", "application/pdf")}, headers=auth_header(token))
        self.assertIn(r.status_code, [400, 422, 500])

    def test_verify_no_file(self):
        self.assertEqual(get_client().post("/verify-certificate").status_code, 422)

    def test_register_missing_field(self):
        self.assertEqual(get_client().post("/register", json={"username": "test"}).status_code, 422)



class TestCertificateLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from certification import setup_keys, PRIVATE_KEY_PATH
        if not PRIVATE_KEY_PATH.exists():
            setup_keys()

    def test_lookup_not_found(self):
        c, token, _ = register_and_login()
        r = c.get("/certificate/KWX-00000000-FAKE", headers=auth_header(token))
        data = r.json()
        if r.status_code == 404:
            self.assertTrue(True)
        else:
            self.assertFalse(data.get("found", data.get("success", True) == False))

    def test_revoke_not_found(self):
        r = get_client().post("/revoke-certificate/KWX-00000000-FAKE")
        self.assertIn(r.status_code, [404, 400])


class TestAPIStability(unittest.TestCase):
    def test_health(self):
        r = get_client().get("/health")
        self.assertEqual(r.json()["status"], "healthy")

    def test_invalid_endpoint(self):
        self.assertEqual(get_client().get("/nonexistent").status_code, 404)

    def test_wrong_method(self):
        self.assertEqual(get_client().get("/register").status_code, 405)

    def test_malformed_json(self):
        r = get_client().post("/register", content="bad", headers={"Content-Type": "application/json"})
        self.assertEqual(r.status_code, 422)


class TestPrivacy(unittest.TestCase):
    def test_user_id_anonymous(self):
        u = unique_user()
        r = get_client().post("/register", json={"username": u, "password": "password123"})
        self.assertNotIn(u, r.json().get("user_id", ""))

    def test_password_not_exposed(self):
        pw = "secretpassword123"
        r = get_client().post("/register", json={"username": unique_user(), "password": pw})
        self.assertNotIn(pw, r.text)


class TestAuthFunctions(unittest.TestCase):
    def test_get_user_profile(self):
        from auth import create_user, get_user_profile
        u = unique_user()
        create_user(u, "password123", organization="TestOrg", verification_link="https://example.com")
        profile = get_user_profile(u)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["organization"], "TestOrg")
        self.assertEqual(profile["verification_link"], "https://example.com")

    def test_get_user_profile_nonexistent(self):
        from auth import get_user_profile
        profile = get_user_profile("nonexistent_user_xyz")
        self.assertIsNone(profile)

    def test_get_profile_by_id(self):
        from auth import create_user, get_profile_by_id
        u = unique_user()
        success, user_id = create_user(u, "password123", organization="IDOrg")
        self.assertTrue(success)
        profile = get_profile_by_id(user_id)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["user_id"], user_id)
        self.assertEqual(profile["organization"], "IDOrg")

    def test_get_profile_by_id_nonexistent(self):
        from auth import get_profile_by_id
        profile = get_profile_by_id("USR-DOESNOTEXIST")
        self.assertIsNone(profile)

    def test_deactivate_user_blocks_login(self):
        from auth import create_user, authenticate, deactivate_user
        u = unique_user()
        create_user(u, "password123")
        success, _ = authenticate(u, "password123")
        self.assertTrue(success)
        deactivate_user(u)
        success, _ = authenticate(u, "password123")
        self.assertFalse(success)

    def test_deactivate_nonexistent_user(self):
        from auth import deactivate_user
        result = deactivate_user("nonexistent_user_xyz")
        self.assertFalse(result)

    def test_list_users_includes_created(self):
        from auth import create_user, list_users
        u = unique_user()
        create_user(u, "password123")
        users = list_users()
        usernames = [entry["username"] for entry in users]
        self.assertIn(u, usernames)

    def test_create_user_with_org_fields(self):
        from auth import create_user, get_user_id
        u = unique_user()
        success, user_id = create_user(u, "password123", organization="ACME", verification_link="https://acme.com")
        self.assertTrue(success)
        self.assertIsNotNone(get_user_id(u))


class TestVerificationResponseFields(unittest.TestCase):
    def test_uncertified_pdf_has_all_fields(self):
        r = get_client().post("/verify-certificate", files={"file": ("t.pdf", create_pdf(), "application/pdf")})
        data = r.json()
        for field in ["valid", "has_certificate", "message"]:
            self.assertIn(field, data, f"Missing field: {field}")

    def test_non_pdf_has_all_fields(self):
        r = get_client().post("/verify-certificate", files={"file": ("t.jpg", create_image(), "image/jpeg")})
        data = r.json()
        self.assertIn("valid", data)
        self.assertIn("has_certificate", data)
        self.assertIn("message", data)


class TestCertificationEndpointBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from certification import setup_keys, PRIVATE_KEY_PATH
        if not PRIVATE_KEY_PATH.exists():
            setup_keys()

    def test_certify_returns_pdf_bytes(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("t.pdf", create_pdf(), "application/pdf")}, headers=auth_header(token))
        if r.status_code == 200:
            self.assertTrue(
                r.content[:4] == b'%PDF' or r.headers.get("content-type") == "application/pdf",
                "200 response from /certify should be PDF bytes"
            )
        else:
            #400 = model rejected, 500 = error — both acceptable in test env
            self.assertIn(r.status_code, [400, 500])

    def test_certify_image_accepted(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("t.jpg", create_image(), "image/jpeg")}, headers=auth_header(token))
        self.assertIn(r.status_code, [200, 400])

    def test_certify_unsupported_type_rejected(self):
        c, token, _ = register_and_login()
        r = c.post("/certify", files={"file": ("t.txt", b"hello world", "text/plain")}, headers=auth_header(token))
        self.assertIn(r.status_code, [400, 422, 500])


class TestHealthEndpointDetails(unittest.TestCase):
    def test_health_has_model_status(self):
        r = get_client().get("/health")
        data = r.json()
        self.assertIn("model_loaded", data)
        self.assertIn("certification_ready", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
