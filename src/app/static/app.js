const form = document.querySelector("#customer-form");
const resultBox = document.querySelector("#result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const customer = Object.fromEntries(formData.entries());

  // Convert form strings into numbers for the API.
  customer.SeniorCitizen = Number(customer.SeniorCitizen);
  customer.tenure = Number(customer.tenure);
  customer.MonthlyCharges = Number(customer.MonthlyCharges);
  customer.TotalCharges = Number(customer.TotalCharges);

  resultBox.hidden = false;
  resultBox.className = "result";
  resultBox.textContent = "Getting prediction…";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(customer),
    });

    const data = await response.json();

    if (!response.ok) {
      const detail = typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail ?? data);
      throw new Error(detail);
    }

    const probability = (data.churn_probability * 100).toFixed(1);
    resultBox.textContent =
      `${data.message}. Estimated churn probability: ${probability}%.`;
  } catch (error) {
    resultBox.className = "result error";
    resultBox.textContent = `Could not get prediction: ${error.message}`;
  }
});