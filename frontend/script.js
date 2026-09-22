const API_BASE = ""; // same-origin: FastAPI serves both the API and this frontend

const EXAMPLES = {
  employees: {
    schema: "CREATE TABLE employees (id INT, name TEXT, department TEXT, salary INT);",
    question: "List the names of employees in the Engineering department earning more than 100000.",
  },
  orders: {
    schema: "CREATE TABLE orders (order_id INT, customer_id INT, amount FLOAT, order_date DATE);",
    question: "What is the total order amount per customer in 2024?",
  },
  movies: {
    schema: "CREATE TABLE movies (title TEXT, year INT, rating FLOAT, genre TEXT);",
    question: "Show the top 5 highest rated horror movies released after 2015.",
  },
};

const schemaEl = document.getElementById("schema");
const questionEl = document.getElementById("question");
const runBtn = document.getElementById("runBtn");
const runBtnLabel = document.getElementById("runBtnLabel");
const sqlOutput = document.getElementById("sqlOutput");
const outputMeta = document.getElementById("outputMeta");
const copyBtn = document.getElementById("copyBtn");
const errorText = document.getElementById("errorText");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const ex = EXAMPLES[btn.dataset.example];
    schemaEl.value = ex.schema;
    questionEl.value = ex.question;
    questionEl.focus();
  });
});

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (res.ok && data.model_loaded) {
      statusDot.className = "status-dot ok";
      statusText.textContent = "model ready";
    } else {
      statusDot.className = "status-dot";
      statusText.textContent = "model loading…";
    }
  } catch {
    statusDot.className = "status-dot down";
    statusText.textContent = "API unreachable";
  }
}

function setLoading(isLoading) {
  runBtn.disabled = isLoading;
  runBtnLabel.textContent = isLoading ? "Generating…" : "Generate SQL";
}

function showError(message) {
  errorText.textContent = message;
  sqlOutput.textContent = "-- generation failed, see error";
  copyBtn.disabled = true;
  outputMeta.textContent = "";
}

function showResult(sql, latencyMs) {
  errorText.textContent = "";
  sqlOutput.textContent = sql || "-- empty response";
  outputMeta.textContent = `${latencyMs.toFixed(0)} ms`;
  copyBtn.disabled = false;
}

async function runQuery() {
  const schema = schemaEl.value.trim();
  const question = questionEl.value.trim();

  if (!schema || !question) {
    showError("Fill in both schema and question first.");
    return;
  }

  setLoading(true);
  errorText.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schema, question }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || `Request failed (${res.status})`);
      return;
    }

    showResult(data.sql, data.latency_ms);
  } catch (err) {
    showError(`Network error: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

runBtn.addEventListener("click", runQuery);

questionEl.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    runQuery();
  }
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(sqlOutput.textContent);
  const original = copyBtn.textContent;
  copyBtn.textContent = "Copied";
  setTimeout(() => { copyBtn.textContent = original; }, 1200);
});

checkHealth();
setInterval(checkHealth, 15000);
