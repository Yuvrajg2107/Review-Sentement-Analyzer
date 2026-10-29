'use strict';
const $ = (s, root = document) => root.querySelector(s);
const examples = {
  positive: 'A beautifully acted film that earns every emotional moment. I loved the patient storytelling, and the final scene stayed with me long after the credits.',
  negative: 'The plot goes nowhere and the dialogue sounds painfully forced. Even the talented cast cannot save this dull, overlong movie. I wish I had skipped it.',
  mixed: 'The performances are excellent and the first half is gripping, but the ending feels rushed. I wanted to love it more than I actually did.'
};
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(id, focus = false) {
  if (!tabs.some(t => t.dataset.tab === id)) id = 'compare';
  tabs.forEach(t => { const selected = t.dataset.tab === id; t.setAttribute('aria-selected', String(selected)); t.tabIndex = selected ? 0 : -1; if (selected && focus) t.focus(); });
  document.querySelectorAll('[role="tabpanel"]').forEach(p => p.hidden = p.id !== id);
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => { history.replaceState(null, '', '#' + t.dataset.tab); selectTab(t.dataset.tab); });
  t.addEventListener('keydown', e => { let next; if (e.key === 'ArrowRight') next = (i + 1) % tabs.length; if (e.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length; if (e.key === 'Home') next = 0; if (e.key === 'End') next = tabs.length - 1; if (next !== undefined) { e.preventDefault(); selectTab(tabs[next].dataset.tab, true); history.replaceState(null, '', '#' + tabs[next].dataset.tab); } });
});
window.addEventListener('hashchange', () => selectTab(location.hash.slice(1)));
selectTab(location.hash.slice(1));
const review = $('#review');
let lastSuccessfulReview = null;
function resetReadings(message = 'Ready to read.') { document.querySelectorAll('.model-result').forEach(card=>{card.classList.remove('positive','negative');$('.verdict',card).textContent=message;$('.confidence',card).textContent='—';$('.probability-track span',card).style.width='0%';$('.probability-label',card).textContent='Positive class probability';$('.latency',card).textContent='—';}); }
function count() { $('#character-count').textContent = `${review.value.length.toLocaleString()} / 12,000`; }
function changedReview(){count();if(lastSuccessfulReview!==null&&review.value.trim()!==lastSuccessfulReview){resetReadings('Review changed.');$('#agreement').textContent='REVIEW CHANGED · RUN AGAIN';}}
review.addEventListener('input', changedReview);
document.querySelectorAll('[data-example]').forEach(b => b.addEventListener('click', () => { review.value = examples[b.dataset.example]; changedReview(); review.focus(); }));
function node(tag, text, className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
function pct(n, digits = 2) { return Number.isFinite(Number(n)) ? (Number(n) * 100).toFixed(digits) + '%' : 'Unavailable'; }
function modelList(report) { if (Array.isArray(report.models)) return report.models; return Object.entries(report.models || {}).map(([id,m]) => ({id,...m})); }
function metric(m, key) { return (m.metrics || m)[key]; }
function identify(m, index) { return /transform|bert/i.test(`${m.id} ${m.name}`) ? 'transformer' : index === 0 ? 'baseline' : 'transformer'; }
async function getJSON(url, options) { const r = await fetch(url, options); let data; try { data = await r.json(); } catch { throw new Error(`The service returned an unreadable response (${r.status}).`); } if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : data.error || 'The model service is unavailable. See the README for setup instructions.'); return data; }
$('#review-form').addEventListener('submit', async e => {
  e.preventDefault(); const text = review.value.trim(); if (!text) { $('#form-status').textContent = 'Add a review first so both models have something to read.'; review.focus(); return; }
  const button = $('#analyze'); button.disabled = true; resetReadings('Reading…'); $('#form-status').classList.remove('error'); $('#form-status').textContent = 'Reading with both models… The first request may need time to load the transformer.'; $('#agreement').textContent = 'READING…';
  try {
    const data = await getJSON('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
    const predictions = data.predictions || [];
    if (predictions.length !== 2) throw new Error('The service did not return both predictions. Please check the server log.');
    predictions.forEach((p,i) => { const card = $('#result-' + identify(p,i)); const positive = String(p.label).toLowerCase() === 'positive'; card.classList.remove('positive','negative'); card.classList.add(positive ? 'positive' : 'negative'); $('.verdict',card).textContent = positive ? 'Positive.' : 'Negative.'; const confidence = $('.confidence',card); confidence.replaceChildren(document.createTextNode(pct(p.confidence,1)), node('small','CONFIDENCE')); const positiveP = p.probabilities?.positive ?? (positive ? p.confidence : 1-p.confidence); $('.probability-track span',card).style.width = `${Math.max(0,Math.min(100,positiveP*100))}%`; $('.probability-label',card).textContent = `${pct(positiveP,1)} positive · ${pct(1-positiveP,1)} negative`; $('.latency',card).textContent = Number.isFinite(Number(p.latency_ms)) ? `${Math.round(p.latency_ms).toLocaleString()} ms` : ''; });
    lastSuccessfulReview=text;
    const agree = predictions[0].label.toLowerCase() === predictions[1].label.toLowerCase(); $('#agreement').textContent = agree ? 'BOTH MODELS AGREE' : 'A DIFFERENCE OF OPINION';
    $('#form-status').textContent = agree ? 'Both models returned the same label. Agreement alone does not verify correctness.' : 'The models disagree. Read the review and decide which label fits better.';
    if (data.disclaimer) $('#prediction-note').textContent = data.disclaimer;
    if(review.value.trim()!==text)changedReview();
  } catch (err) { resetReadings('Unavailable.'); $('#form-status').classList.add('error'); $('#form-status').textContent = err.message; $('#agreement').textContent = 'REQUEST NOT COMPLETED'; }
  finally { button.disabled = false; }
});
async function loadStatus() { try { const data = await getJSON('/api/status'); const ready = data.ready ?? data.baseline_ready ?? data.models_ready; $('#service-status').replaceChildren(node('i'),document.createTextNode(ready === false ? 'Model setup in progress' : 'Local model service connected')); } catch { $('#service-status').replaceChildren(node('i'),document.createTextNode('Model service unavailable')); } }
function renderMatrix(m) { const matrix = metric(m,'confusion_matrix'); if (!Array.isArray(matrix) || !Array.isArray(matrix[0])) return null; const card = node('article',undefined,'confusion-card'); card.append(node('h3',m.name || m.id),node('p','Confusion matrix · rows = actual, columns = predicted')); const grid = node('div',undefined,'matrix'); grid.append(node('span','', 'axis'),node('span','Pred. negative','axis'),node('span','Pred. positive','axis')); ['Actual negative','Actual positive'].forEach((label,r) => { grid.append(node('span',label,'axis')); for(let c=0;c<2;c++){ const cell = node('div',Number(matrix[r][c]).toLocaleString(),'cell'+(r===c?' correct':'')); cell.append(node('small',r===c?'CORRECT':'INCORRECT')); grid.append(cell); } }); card.append(grid,node('div','Dark cells are correct predictions; light cells are errors.','matrix-note')); return card; }
async function loadReport() {
  try {
    const report = await getJSON('/api/report'); const models = modelList(report); if (!models.length) throw new Error('No evaluated models are present in the saved report.');
    $('#report-status').hidden = true; $('#report-content').hidden = false;
    ['#dataset-strip','#metrics-body','#confusion-grid'].forEach(selector=>$(selector).replaceChildren());
    const ds = report.dataset || {}; const strip = $('#dataset-strip');
    const items = [['DATASET',typeof ds === 'string' ? ds : ds.name || ds.id || ds.repo_id || 'IMDb'],['EVALUATION',ds.test_size || ds.test_count || report.test_size || report.test_count || 'See saved report'],['LABELS','Negative / Positive']];
    items.forEach(([label,value])=> { const el=node('div'); el.append(node('small',label),document.createTextNode(typeof value === 'number' ? value.toLocaleString()+' reviews' : value)); strip.append(el); });
    models.forEach(m => { const tr=node('tr'); tr.append(node('td',m.name || m.id)); ['accuracy','precision','recall','f1'].forEach(k => tr.append(node('td',pct(metric(m,k))))); $('#metrics-body').append(tr); const matrix=renderMatrix(m); if(matrix)$('#confusion-grid').append(matrix); if(identify(m,models.indexOf(m))==='transformer') $('#transformer-name').textContent=m.model_id || m.checkpoint || m.name || 'DistilBERT · IMDb'; });
    const ranked = [...models].filter(m=>Number.isFinite(Number(metric(m,'accuracy')))).sort((a,b)=>metric(b,'accuracy')-metric(a,'accuracy'));
    if(ranked.length>1) { const difference=(metric(ranked[0],'accuracy')-metric(ranked[1],'accuracy'))*100; $('#report-insight').textContent = difference===0 ? 'Both models achieved the same accuracy on this test set.' : `${ranked[0].name || ranked[0].id} leads by ${difference.toFixed(2)} percentage points in test accuracy.`; }
    const training=report.training || {}; const selected=training.selected || training.selected_params || training.best_params || training.configuration || training.best_config || {}; const summary=$('#training-summary'); summary.replaceChildren();
    const fields = [['Random seed',training.seed],['Selected features',selected.features],['Regularization C',selected.C],['Final training reviews',training.refit_train_count],['Transformer token limit',training.transformer_max_length],['Inference device',training.transformer_device]];
    fields.filter(([,value])=>value!==undefined).forEach(([label,value])=>{const row=node('div',undefined,'run-field');row.append(node('span',label),node('strong',String(value)));summary.append(row);});
    if(Array.isArray(training.candidates)){
      const wrap=node('div',undefined,'table-wrap');const table=node('table');table.append(node('caption','Train-only validation · candidate selection by positive-class F1'));
      const header=node('tr');['Features / C','Accuracy','F1'].forEach(value=>header.append(node('th',value)));const head=node('thead');head.append(header);table.append(head);
      const body=node('tbody');training.candidates.forEach(candidate=>{const row=node('tr');row.append(node('td',`${candidate.features} / ${candidate.C}`),node('td',pct(candidate.accuracy)),node('td',pct(candidate.f1)));body.append(row);});table.append(body);wrap.append(table);summary.append(wrap);
    }
    const selectionText=selected.features ? `The selected baseline uses ${selected.features==='word_char'?'combined word and character':'word-only'} features with regularization C = ${selected.C}.` : 'The selected baseline configuration is recorded in the metrics JSON.';
    if(selected.features)$('#baseline-feature-name').textContent=/char/.test(selected.features)?'Word phrases + character patterns':'Word unigrams + bigrams';
    $('#architecture-details').textContent=selectionText+(training.refit_train_count ? ` After validation, the selected pipeline was refitted on all ${Number(training.refit_train_count).toLocaleString()} official training reviews.` : '');
  } catch(err) { $('#report-status').textContent=`Evaluation results are not available yet. ${err.message}`; }
}
$('#training-image').addEventListener('error',e=>{e.target.hidden=true;$('#image-missing').hidden=false;});
async function loadTrainingLog(){try{const r=await fetch('/api/training-log');if(!r.ok)throw new Error();$('#training-log').textContent=await r.text();$('#log-state').textContent='SAVED RUN OUTPUT';}catch{$('#training-log').textContent='No training log is available yet. Start the experiment using the README instructions.';$('#log-state').textContent='AWAITING TRAINING';}}
loadStatus(); loadReport(); loadTrainingLog();
// Artifacts may finish while a visitor is reading another tab.
tabs.forEach(tab=>tab.addEventListener('click',()=>{loadStatus();if(tab.dataset.tab==='reports')loadReport();if(tab.dataset.tab==='evidence')loadTrainingLog();}));
async function loadFeatureWeights(){
  try{
    const weights=await getJSON('/api/features');
    const details=node('details',undefined,'learned-weights');details.append(node('summary','Inspect the learned feature weights'));
    details.append(node('p',weights.note || 'Global fitted coefficients; feature values also affect each prediction.'));
    details.append(node('p','word__ marks a word feature; char__ marks a character feature. Positive coefficients support the positive class. Negative coefficients support the negative class.'));
    const grid=node('div',undefined,'coefficient-grid');
    [['negative','Negative evidence'],['positive','Positive evidence']].forEach(([key,title])=>{
      const table=node('table');table.append(node('caption',title));const head=node('thead');const header=node('tr');header.append(node('th','Feature'),node('th','Coefficient'));head.append(header);table.append(head);
      const body=node('tbody');(weights[key] || []).slice(0,10).forEach(item=>{const row=node('tr');row.append(node('td',item.feature),node('td',Number(item.coefficient).toFixed(3)));body.append(row);});table.append(body);grid.append(table);
    });details.append(grid);$('#architecture').append(details);
  }catch{/* The static architecture remains useful before training finishes. */}
}
loadFeatureWeights();
