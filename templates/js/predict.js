async function predict() {
  const data = {
    age: parseFloat(age.value),
    gender: parseInt(gender.value),
    height: parseFloat(height.value),
    weight: parseFloat(weight.value),
    ap_hi: parseFloat(ap_hi.value),
    ap_lo: parseFloat(ap_lo.value),
    smoke: 0,
    alco: 0,
    active: 1,
    cholesterol_2: 0,
    cholesterol_3: 0,
    gluc_2: 0,
    gluc_3: 0
  };

  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });

  const r = await res.json();

  document.getElementById("result").innerHTML = `
    <h2 class="text-2xl font-bold mb-2">${r.prediction}</h2>
    <p class="text-lg">Risk Probability: ${(r.probability * 100).toFixed(2)}%</p>
  `;
  document.getElementById("result").classList.remove("hidden");
}
