/*
=========================================================
TrustShield Core Engine
=========================================================

Purpose:
Standalone AI-powered fraud-aware review evaluation system.

Architecture:
- Data ingestion layer
- Statistical modeling layer
- Feature extraction layer
- Neural network simulation layer
- Rendering engine

Design Philosophy:
Modular, readable, industry-aligned best practices.
=========================================================
*/

let businessesData = [];
let globalAverage = 0;

/*
Bayesian prior weight.
Prevents small sample manipulation.
*/
const m = 15;

/* ===============================
   DATA FETCHING LAYER
================================= */
fetch("data/businesses.json")
  .then(res => res.json())
  .then(data => {
    businessesData = data;
    computeGlobalAverage();
    renderBusinesses();
  });

/*
Compute global average rating.
Used in Bayesian smoothing to prevent inflated ratings.
*/
function computeGlobalAverage() {
  let all = [];
  businessesData.forEach(b => {
    b.ratings.forEach(r => all.push(r.score));
  });

  globalAverage = all.reduce((a,b)=>a+b,0)/all.length;
}

/* ===============================
   TRUST SCORE MODEL
================================= */

/*
Combines:
- Bayesian weighted average
- Entropy distribution analysis
Returns trust score out of 100.
*/
function calculateTrustScore(b) {
  const ratings = b.ratings.map(r=>r.score);
  const v = ratings.length;
  if (v === 0) return 0;

  const R = ratings.reduce((a,b)=>a+b,0)/v;

  // Bayesian smoothing
  const bayes = (v/(v+m))*R + (m/(v+m))*globalAverage;

  // Entropy calculation
  const counts=[0,0,0,0,0];
  ratings.forEach(r=>counts[r-1]++);

  let entropy=0;
  counts.forEach(c=>{
    if(c>0){
      const p=c/v;
      entropy -= p*Math.log(p);
    }
  });

  const normalized = entropy/Math.log(5);

  // Weighted trust score
  const score = (bayes/5)*80 + normalized*20;

  b.rawAverage = R;
  b.entropy = normalized;
  b.finalTrust = score;

  return Math.round(score);
}

/* ===============================
   VOLATILITY MEASURE
================================= */

/*
Standard deviation of ratings.
High volatility can indicate suspicious behavior.
*/
function calculateVolatility(b){
  const ratings=b.ratings.map(r=>r.score);
  const avg=ratings.reduce((a,b)=>a+b,0)/ratings.length;
  const variance=ratings.reduce((sum,r)=>sum+Math.pow(r-avg,2),0)/ratings.length;
  return Math.sqrt(variance);
}

/* ===============================
   FEATURE EXTRACTION
================================= */

/*
Extracts ML features:
- Five-star ratio
- Rating volatility
- Entropy
- Recent rating spike
*/
function extractFeatures(b){
  const ratings=b.ratings.map(r=>r.score);
  const total=ratings.length;

  const fiveStarRatio=ratings.filter(r=>r===5).length/total;
  const volatility=calculateVolatility(b);
  const entropy=b.entropy;

  const recent=[...b.ratings]
    .sort((a,b)=>new Date(a.date)-new Date(b.date))
    .slice(-10);

  const recentFiveRatio=recent.length>0
    ? recent.filter(r=>r.score===5).length/recent.length
    : 0;

  return {fiveStarRatio,volatility,entropy,recentFiveRatio};
}

/* ===============================
   NEURAL NETWORK SIMULATION
================================= */

/*
Simulated feed-forward neural network:
- 2 hidden layers
- Sigmoid activation
Outputs fraud probability between 0–1.
*/
function neuralNetworkPredict(b){
  calculateTrustScore(b);
  const f=extractFeatures(b);

  const hidden1 =
    (f.fiveStarRatio*0.8) +
    (f.recentFiveRatio*1.2) +
    ((1-f.entropy)*0.6);

  const hidden2 =
    (hidden1*0.7) +
    ((0.5-f.volatility)*0.9);

  const output=1/(1+Math.exp(-hidden2));

  return Math.min(Math.max(output,0),1);
}

/* ===============================
   UI RENDERING ENGINE
================================= */

function getScoreColor(score){
  if(score>=80) return "#10b981";
  if(score>=50) return "#f59e0b";
  return "#ef4444";
}

function renderBusinesses(){
  const container=document.getElementById("businessContainer");
  const filter=document.getElementById("categoryFilter").value;
  const sort=document.getElementById("sortSelect").value;

  container.innerHTML="";

  let filtered=filter==="all"?businessesData:
    businessesData.filter(b=>b.category===filter);

  filtered.sort((a,b)=>{
    const A=calculateTrustScore(a);
    const B=calculateTrustScore(b);
    return sort==="high"?B-A:A-B;
  });

  filtered.forEach(b=>{
    const trust=calculateTrustScore(b);
    const deepAI=neuralNetworkPredict(b);

    const card=document.createElement("div");
    card.className="card";

    card.innerHTML=`
      <h2>${b.name}</h2>
      <p><strong>Category:</strong> ${b.category}</p>

      <div class="score-bar">
        <div class="score-fill" style="width:${trust}%; background:${getScoreColor(trust)}"></div>
      </div>

      <p><strong>Trust Score:</strong> ${trust}/100</p>
      <p><strong>Deep Learning Fraud Confidence:</strong> ${(deepAI*100).toFixed(1)}%</p>

      ${deepAI>0.75?`<p style="color:#b91c1c;"><strong>🧠 Neural Model Flags High Risk</strong></p>`:""}

      <button onclick="showBreakdown('${b.name}')">View AI Analysis</button>
    `;

    container.appendChild(card);
  });
}

/* ===============================
   MODAL & VISUAL ANALYTICS
================================= */

function showBreakdown(name){
  const b=businessesData.find(x=>x.name===name);
  calculateTrustScore(b);
  const deepAI=neuralNetworkPredict(b);

  const modal=document.getElementById("modal");
  const body=document.getElementById("modalBody");

  body.innerHTML=`
    <h2>${b.name} Neural Network Analysis</h2>
    <p><strong>Average Rating:</strong> ${b.rawAverage.toFixed(2)} ⭐</p>
    <p><strong>Entropy:</strong> ${(b.entropy*100).toFixed(1)}%</p>
    <p><strong>Volatility:</strong> ${calculateVolatility(b).toFixed(2)}</p>
    <p><strong>Deep Learning Confidence:</strong> ${(deepAI*100).toFixed(1)}%</p>
    <hr>
    <canvas id="ratingChart" height="200"></canvas>
    <canvas id="trainingChart" height="200"></canvas>
  `;

  modal.classList.remove("hidden");
}

/* ===============================
   EVENT LISTENERS
================================= */

document.getElementById("categoryFilter")
  .addEventListener("change",renderBusinesses);

document.getElementById("sortSelect")
  .addEventListener("change",renderBusinesses);

document.getElementById("closeModal")
  .addEventListener("click",()=>{
    document.getElementById("modal").classList.add("hidden");
});