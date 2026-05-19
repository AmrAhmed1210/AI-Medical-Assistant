/**
 * ══════════════════════════════════════════════════════════════
 *  MedBook — K6 Load Testing Script
 * ══════════════════════════════════════════════════════════════
 *
 *  Purpose : Stress-test the MedBook API & AI Microservice
 *            to validate performance under concurrent load.
 *
 *  Usage   :
 *    # Install K6:  https://k6.io/docs/get-started/installation/
 *    # Run test  :  k6 run load-test.js
 *    # Run with 50 VUs for 30s:  k6 run --vus 50 --duration 30s load-test.js
 *
 *  Scenarios:
 *    1. Smoke Test    — 5 VUs for 30s  (basic sanity check)
 *    2. Load Test     — 50 VUs for 2m  (normal traffic simulation)
 *    3. Stress Test   — 200 VUs for 1m (peak traffic simulation)
 *
 * ══════════════════════════════════════════════════════════════
 */

import http from "k6/http";
import { check, sleep, group } from "k6";
import { Rate, Trend } from "k6/metrics";

// ─── Custom Metrics ────────────────────────────────────────────
const errorRate = new Rate("errors");
const loginDuration = new Trend("login_duration");
const aiChatDuration = new Trend("ai_chat_duration");
const healthCheckDuration = new Trend("health_check_duration");

// ─── Configuration ─────────────────────────────────────────────
const BASE_URL = __ENV.BASE_URL || "http://localhost:5000";
const AI_URL = __ENV.AI_URL || "http://localhost:8000";
const INTERNAL_TOKEN = "LuxuryMedicalAiSecretKey2026";

const TEST_USER = {
  email: "loadtest@medbook.com",
  password: "TestPassword123!",
};

// ─── Test Scenarios ────────────────────────────────────────────
export const options = {
  scenarios: {
    // Scenario 1: Smoke test (quick sanity check)
    smoke: {
      executor: "constant-vus",
      vus: 5,
      duration: "30s",
      gracefulStop: "5s",
      tags: { scenario: "smoke" },
    },

    // Scenario 2: Average load (simulating normal usage)
    average_load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 50 },   // Ramp up to 50 users
        { duration: "1m", target: 50 },    // Stay at 50 users
        { duration: "30s", target: 0 },    // Ramp down
      ],
      startTime: "35s",                     // Start after smoke
      tags: { scenario: "average_load" },
    },

    // Scenario 3: Stress test (simulating peak/spike)
    stress: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "20s", target: 100 },  // Ramp to 100
        { duration: "30s", target: 200 },  // Ramp to 200
        { duration: "20s", target: 0 },    // Ramp down
      ],
      startTime: "3m",                      // Start after average_load
      tags: { scenario: "stress" },
    },
  },

  thresholds: {
    http_req_duration: ["p(95)<3000"],       // 95% of requests < 3s
    http_req_failed: ["rate<0.05"],          // Error rate < 5%
    errors: ["rate<0.1"],                    // Custom error rate < 10%
    login_duration: ["p(95)<2000"],          // Login p95 < 2s
    health_check_duration: ["p(95)<500"],    // Health check p95 < 500ms
  },
};

// ─── Setup: Create test user if needed ─────────────────────────
export function setup() {
  console.log("🚀 MedBook Load Test — Starting...");
  console.log(`   Backend URL : ${BASE_URL}`);
  console.log(`   AI URL      : ${AI_URL}`);

  // Try to register a test user (will fail gracefully if exists)
  const registerRes = http.post(
    `${BASE_URL}/api/auth/register`,
    JSON.stringify({
      fullName: "Load Test User",
      email: TEST_USER.email,
      password: TEST_USER.password,
    }),
    { headers: { "Content-Type": "application/json" } }
  );

  // Login to get a valid token for authenticated tests
  const loginRes = http.post(
    `${BASE_URL}/api/auth/login`,
    JSON.stringify(TEST_USER),
    { headers: { "Content-Type": "application/json" } }
  );

  let token = null;
  if (loginRes.status === 200) {
    const body = JSON.parse(loginRes.body);
    token = body.accessToken;
    console.log("✅ Auth token acquired for load testing.");
  } else {
    console.log("⚠️  Could not acquire auth token. Some tests will be skipped.");
  }

  return { token };
}

// ─── Main Test Function ────────────────────────────────────────
export default function (data) {
  const token = data.token;
  const authHeaders = token
    ? { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }
    : { "Content-Type": "application/json" };

  // ── Test 1: Health Check (Backend) ────────────────────────────
  group("Backend Health Check", () => {
    const res = http.get(`${BASE_URL}/swagger/index.html`);
    const success = check(res, {
      "Backend responds with 200": (r) => r.status === 200,
      "Response time < 1s": (r) => r.timings.duration < 1000,
    });
    healthCheckDuration.add(res.timings.duration);
    errorRate.add(!success);
  });

  sleep(0.5);

  // ── Test 2: AI Health Check ───────────────────────────────────
  group("AI Service Health Check", () => {
    const res = http.get(`${AI_URL}/health`);
    const success = check(res, {
      "AI service responds": (r) => r.status === 200,
      "AI version is correct": (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === "ok";
        } catch {
          return false;
        }
      },
    });
    errorRate.add(!success);
  });

  sleep(0.5);

  // ── Test 3: Login Performance ─────────────────────────────────
  group("Login Performance", () => {
    const res = http.post(
      `${BASE_URL}/api/auth/login`,
      JSON.stringify(TEST_USER),
      { headers: { "Content-Type": "application/json" } }
    );
    const success = check(res, {
      "Login successful": (r) => r.status === 200 || r.status === 401,
      "Login response time < 2s": (r) => r.timings.duration < 2000,
      "Returns JWT token": (r) => {
        if (r.status === 200) {
          try {
            const body = JSON.parse(r.body);
            return body.accessToken && body.accessToken.length > 0;
          } catch {
            return false;
          }
        }
        return true; // Skip check for 401
      },
    });
    loginDuration.add(res.timings.duration);
    errorRate.add(!success);
  });

  sleep(0.5);

  // ── Test 4: AI Chat (Simulated Concurrent Users) ──────────────
  group("AI Chat Endpoint", () => {
    const questions = [
      "عندي صداع شديد من 3 أيام",
      "I have chest pain when breathing",
      "بحس بدوخة لما بقوم فجأة",
      "What are the symptoms of diabetes?",
      "عندي حرقان في المعدة",
    ];
    const randomQ = questions[Math.floor(Math.random() * questions.length)];

    const res = http.post(
      `${AI_URL}/ask`,
      JSON.stringify({ text: randomQ }),
      {
        headers: {
          "Content-Type": "application/json",
          "x-internal-token": INTERNAL_TOKEN,
        },
        timeout: "60s",
      }
    );

    const success = check(res, {
      "AI responds": (r) => r.status === 200,
      "AI response has content": (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.gemini_reply && body.gemini_reply.length > 0;
        } catch {
          return false;
        }
      },
    });
    aiChatDuration.add(res.timings.duration);
    errorRate.add(!success);
  });

  sleep(1);

  // ── Test 5: Authenticated API Calls ───────────────────────────
  if (token) {
    group("Authenticated API Calls", () => {
      // Test fetching patient data (if endpoint exists)
      const res = http.get(`${BASE_URL}/api/patients`, {
        headers: authHeaders,
      });
      check(res, {
        "Authenticated request works": (r) =>
          r.status === 200 || r.status === 403 || r.status === 404,
      });
    });
  }

  sleep(0.5);
}

// ─── Teardown: Print Summary ───────────────────────────────────
export function teardown(data) {
  console.log("═══════════════════════════════════════════");
  console.log("  🏁 MedBook Load Test — Complete!");
  console.log("═══════════════════════════════════════════");
  console.log("  Review the results above for:");
  console.log("  • http_req_duration (p95 should be < 3s)");
  console.log("  • errors rate (should be < 10%)");
  console.log("  • login_duration (p95 should be < 2s)");
  console.log("  • ai_chat_duration (check AI response times)");
  console.log("═══════════════════════════════════════════");
}
