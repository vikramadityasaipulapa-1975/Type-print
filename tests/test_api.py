from __future__ import annotations

import base64
import asyncio
import os
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.core.config import settings
from app.database.session import Base, get_db
from app.main import app
from app.models.entities import AuthenticationLog, TypingSession
import app.services.biometric as biometric_service


def event_payload(count: int = 16) -> list[dict[str, object]]:
    events = []
    down = 0.0
    for index in range(count):
        code = "Space" if index % 6 == 5 else f"Key{chr(65 + index % 26)}"
        dwell = 55.0 + index % 4
        flight = None if index == 0 else 18.0 + index % 3
        events.append({"id": index, "code": code, "downMs": down, "upMs": down + dwell, "dwellMs": dwell, "flightMs": flight})
        down += dwell + (flight or 0.0)
    return events


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.SessionTest = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

        def override_db():
            db = self.SessionTest()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_db
        self.jwt_patch = patch.object(settings, "jwt_secret_key", "unit-test-secret-key-with-more-than-32-characters")
        self.encryption_patch = patch.object(settings, "biometric_encryption_key", base64.urlsafe_b64encode(os.urandom(32)).decode("ascii"))
        self.jwt_patch.start()
        self.encryption_patch.start()
        biometric_service._model_service = None
        self.client = TestClient(app)
        self.email = f"user-{uuid4().hex[:10]}@example.com"
        self.username = f"user_{uuid4().hex[:10]}"

    def tearDown(self) -> None:
        self.client.close()
        app.dependency_overrides.clear()
        biometric_service._model_service = None
        self.jwt_patch.stop()
        self.encryption_patch.stop()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def register_and_login(self) -> tuple[str, int]:
        response = self.client.post("/api/auth/register", json={"username": self.username, "email": self.email, "password": "A-real-password-123!"})
        self.assertEqual(response.status_code, 201, response.text)
        login = self.client.post("/api/auth/login", json={"identifier": self.email, "password": "A-real-password-123!"})
        self.assertEqual(login.status_code, 200, login.text)
        return login.json()["access_token"], response.json()["id"]

    def test_auth_register_login_me_logout_and_revocation(self) -> None:
        token, user_id = self.register_and_login()
        headers = {"Authorization": f"Bearer {token}"}
        me = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["id"], user_id)
        logout = self.client.post("/api/auth/logout", headers=headers)
        self.assertEqual(logout.status_code, 204)
        self.assertEqual(self.client.get("/api/auth/me", headers=headers).status_code, 401)

    def test_analysis_enrollment_analytics_and_biometric_deletion(self) -> None:
        token, _ = self.register_and_login()
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "sample_id": str(uuid4()),
            "session_id": str(uuid4()),
            "consent": True,
            "passage_completed": True,
            "events": event_payload(),
        }
        analysis = self.client.post("/api/typing/analyze", headers=headers, json=payload)
        self.assertEqual(analysis.status_code, 200, analysis.text)
        self.assertGreater(analysis.json()["average_dwell_ms"], 0)
        self.assertEqual(self.client.post("/api/biometrics/enroll", headers=headers, json={**payload, "consent": False}).status_code, 403)

        first = self.client.post("/api/biometrics/enroll", headers=headers, json=payload)
        self.assertEqual(first.status_code, 201, first.text)
        self.assertEqual(first.json()["status"], "collecting")
        self.assertFalse(first.json()["profile_ready"])

        profile = self.client.get("/api/biometrics/profile", headers=headers)
        self.assertEqual(profile.status_code, 200, profile.text)
        self.assertEqual(profile.json()["enrollment_samples"], 1)
        self.assertEqual(self.client.post("/api/typing/authenticate", headers=headers, json={**payload, "sample_id": str(uuid4())}).status_code, 409)

        history = self.client.get("/api/analytics/history", headers=headers)
        timing = self.client.get("/api/analytics/timing", headers=headers)
        model_status = self.client.get("/api/model/status", headers=headers)
        model_metrics = self.client.get("/api/model/metrics", headers=headers)
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(timing.status_code, 200, timing.text)
        self.assertEqual(len(timing.json()[0]["sequence"]), 16)
        self.assertEqual(model_status.status_code, 200, model_status.text)
        self.assertFalse(model_status.json()["verification_ready"])
        self.assertEqual(model_metrics.status_code, 200, model_metrics.text)

        with self.SessionTest() as db:
            row = db.scalar(select(TypingSession).where(TypingSession.session_type == "enrollment"))
            self.assertIsNotNone(row)
            self.assertNotIn("KeyA", row.sample_metadata)
            self.assertEqual(db.scalar(select(AuthenticationLog)), None)

        dashboard = self.client.get("/api/analytics/dashboard", headers=headers)
        self.assertEqual(dashboard.status_code, 200, dashboard.text)
        self.assertEqual(dashboard.json()["enrollment_samples"], 1)
        self.assertGreater(dashboard.json()["typing_speed_cpm"], 0)
        deleted = self.client.delete("/api/biometrics/profile", headers=headers)
        self.assertEqual(deleted.status_code, 204)
        self.assertFalse(self.client.get("/api/biometrics/profile", headers=headers).json()["enrolled"])
        with self.SessionTest() as db:
            self.assertEqual(db.scalar(select(TypingSession)), None)

    def test_protected_endpoints_require_a_bearer_session(self) -> None:
        response = self.client.get("/api/analytics/dashboard")
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
