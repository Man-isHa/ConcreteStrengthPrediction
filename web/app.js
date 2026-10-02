const form = document.querySelector("#prediction-form");
const materialButtons = [...document.querySelectorAll(".material-option")];
const resultMessages = document.querySelector("#result-messages");
const progressFill = document.querySelector("#progress-fill");
const progressCopy = document.querySelector("#progress-copy");
const progressBar = document.querySelector(".progress-track");
const errorMessage = document.querySelector("#form-error");
const submitButton = document.querySelector(".submit-button");
let selectedMaterial = "RHA";

materialButtons.forEach((button) => {
  button.addEventListener("click", () => {
    selectedMaterial = button.dataset.material;
    materialButtons.forEach((option) => {
      const selected = option === button;
      option.classList.toggle("selected", selected);
      option.setAttribute("aria-pressed", String(selected));
      option.querySelector(".material-check").textContent = selected ? "Selected" : "Select";
    });
    updateProgress();
  });
});

function updateProgress() {
  const required = ["replacement_pct", "age_days", "ctrl_strength"];
  const completed = required.filter((name) => form.elements[name].value.trim() !== "").length;
  progressFill.style.width = `${(completed / required.length) * 100}%`;
  progressCopy.textContent = `${completed} of ${required.length} complete`;
  progressBar.setAttribute("aria-valuenow", String(completed));
}

form.addEventListener("input", (event) => {
  if (event.target.matches("input")) updateProgress();
});

function appendMessage({ kind, label, text, result, error }) {
  const article = document.createElement("article");
  article.className = "message";
  const avatar = document.createElement("div");
  avatar.className = `avatar ${kind === "user" ? "user-avatar" : "assistant-avatar"}`;
  avatar.textContent = kind === "user" ? "Y" : "S";
  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const name = document.createElement("strong");
  name.textContent = kind === "user" ? "You" : "Strength Desk";
  const badge = document.createElement("span");
  badge.textContent = kind === "user" ? "INPUT" : "MODEL RESPONSE";
  meta.append(name, badge);
  const bubble = document.createElement("div");
  bubble.className = `bubble ${kind === "user" ? "user-bubble" : "result-bubble"}${error ? " error-bubble" : ""}`;

  if (result) {
    const top = document.createElement("div");
    top.className = "result-top";
    const valueGroup = document.createElement("div");
    const resultLabel = document.createElement("div");
    resultLabel.className = "result-label";
    resultLabel.textContent = "PREDICTED COMPRESSIVE STRENGTH";
    const value = document.createElement("div");
    value.className = "result-value";
    value.append(document.createTextNode(result.strength_mpa.toFixed(2) + " "));
    const unit = document.createElement("small");
    unit.textContent = "MPa";
    value.append(unit);
    valueGroup.append(resultLabel, value);
    const tag = document.createElement("span");
    tag.className = "material-tag";
    tag.textContent = result.material;
    top.append(valueGroup, tag);
    const model = document.createElement("div");
    model.className = "result-model";
    model.append(document.createTextNode("Model: "));
    const modelName = document.createElement("span");
    modelName.textContent = result.model;
    model.append(modelName, document.createTextNode(` · typical held-out MAE ${result.expected_mae_mpa} MPa`));
    const inputSummary = document.createElement("div");
    inputSummary.className = "used-inputs";
    inputSummary.textContent = `Inputs used: ${result.inputs.replacement_pct}% by mass of cement · ${result.inputs.age_days} days · ${result.inputs.ctrl_strength} MPa control mixture strength`;
    bubble.append(top, model, inputSummary);
    const notes = document.createElement("div");
    notes.className = "result-notes";
    [...result.assumptions, ...result.warnings].forEach((note, index) => {
      const line = document.createElement("div");
      line.className = `note-line${index >= result.assumptions.length ? " warning" : ""}`;
      line.textContent = `${index < result.assumptions.length ? "Assumption: " : "Range note: "}${note}`;
      notes.append(line);
    });
    if (notes.childElementCount) bubble.append(notes);
  } else {
    bubble.textContent = error || text || label;
  }
  body.append(meta, bubble);
  article.append(avatar, body);
  resultMessages.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  errorMessage.textContent = "";
  submitButton.disabled = true;
  submitButton.querySelector("span").textContent = "Running model...";

  const inputs = Object.fromEntries(new FormData(form).entries());
  const typedInputs = Object.entries(inputs).filter(([, value]) => value.trim() !== "");
  const summary = typedInputs.length
    ? `${selectedMaterial} mix · ${typedInputs.length} parameter(s) entered`
    : "No parameters entered · use example assumptions";
  appendMessage({ kind: "user", text: summary });

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ material: selectedMaterial, inputs }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The prediction request failed.");
    appendMessage({ kind: "assistant", result: data });
  } catch (error) {
    appendMessage({ kind: "assistant", error: error.message });
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span").textContent = "Predict strength";
  }
});

updateProgress();