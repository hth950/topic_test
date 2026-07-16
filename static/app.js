const state = {
  config: null,
  folders: [],
  selectedFolderId: null,
  images: [],
  runs: [],
  auditReports: [],
  jobs: {},
  pollers: {},
  selectedImage: null,
  currentRun: null,
  activeSource: "chandra",
  activeExperiment: null,
  imageFilter: "",
};

const el = {
  configStatus: document.querySelector("#configStatus"),
  folderSelect: document.querySelector("#folderSelect"),
  folderList: document.querySelector("#folderList"),
  folderSummary: document.querySelector("#folderSummary"),
  imageSummary: document.querySelector("#imageSummary"),
  imageSearch: document.querySelector("#imageSearch"),
  pageJump: document.querySelector("#pageJump"),
  imagePager: document.querySelector("#imagePager"),
  prevImageBtn: document.querySelector("#prevImageBtn"),
  nextImageBtn: document.querySelector("#nextImageBtn"),
  imageList: document.querySelector("#imageList"),
  runList: document.querySelector("#runList"),
  auditReportList: document.querySelector("#auditReportList"),
  imageTitle: document.querySelector("#imageTitle"),
  runStatus: document.querySelector("#runStatus"),
  userConditions: document.querySelector("#userConditions"),
  sourceImage: document.querySelector("#sourceImage"),
  cropBox: document.querySelector("#cropBox"),
  bboxX: document.querySelector("#bboxX"),
  bboxY: document.querySelector("#bboxY"),
  bboxW: document.querySelector("#bboxW"),
  bboxH: document.querySelector("#bboxH"),
  cropPreview: document.querySelector("#cropPreview"),
  maskedPreview: document.querySelector("#maskedPreview"),
  matrixEditor: document.querySelector("#matrixEditor"),
  htmlPreview: document.querySelector("#htmlPreview"),
  rawOutput: document.querySelector("#rawOutput"),
  traceSummary: document.querySelector("#traceSummary"),
  traceInput: document.querySelector("#traceInput"),
  tracePrompt: document.querySelector("#tracePrompt"),
  traceRequest: document.querySelector("#traceRequest"),
  validationStatus: document.querySelector("#validationStatus"),
  finalLinks: document.querySelector("#finalLinks"),
  experimentProvider: document.querySelector("#experimentProvider"),
  runExperimentsBtn: document.querySelector("#runExperimentsBtn"),
  experimentJobPanel: document.querySelector("#experimentJobPanel"),
  experimentJobState: document.querySelector("#experimentJobState"),
  experimentJobTarget: document.querySelector("#experimentJobTarget"),
  experimentProgressFill: document.querySelector("#experimentProgressFill"),
  experimentProgressText: document.querySelector("#experimentProgressText"),
  experimentJobStrategy: document.querySelector("#experimentJobStrategy"),
  experimentResults: document.querySelector("#experimentResults"),
};

async function boot() {
  bindEvents();
  renderEmptyGrid();
  state.config = await apiGet("/api/config");
  renderConfig();
  const folderData = await apiGet("/api/image-folders");
  state.folders = folderData.folders || [];
  const params = new URLSearchParams(window.location.search);
  const requestedFolderId = params.get("folder_id");
  const requestedRunId = params.get("run_id");
  const requestedFolderExists = state.folders.some((folder) => folder.id === requestedFolderId);
  state.selectedFolderId = requestedFolderExists ? requestedFolderId : folderData.default_folder_id || state.folders[0]?.id || null;
  renderFolders();
  await refreshAuditReports();
  await loadFolder(state.selectedFolderId, { selectFirst: !requestedRunId });
  if (requestedRunId) {
    await loadRun(requestedRunId);
  }
}

function bindEvents() {
  document.querySelector("#autoCropBtn").addEventListener("click", () => cropSelected(null));
  document.querySelector("#manualCropBtn").addEventListener("click", () => cropSelected(readBboxForm()));
  document.querySelector("#refreshRunBtn").addEventListener("click", refreshRun);
  document.querySelector("#refreshRunsBtn").addEventListener("click", refreshRuns);
  document.querySelector("#refreshReportsBtn").addEventListener("click", refreshAuditReports);
  el.folderSelect.addEventListener("change", () => loadFolder(el.folderSelect.value));
  el.imageSearch.addEventListener("input", () => {
    state.imageFilter = el.imageSearch.value.trim().toLowerCase();
    renderImages();
  });
  el.prevImageBtn.addEventListener("click", () => selectAdjacentImage(-1));
  el.nextImageBtn.addEventListener("click", () => selectAdjacentImage(1));
  el.pageJump.addEventListener("change", () => selectImageByVisibleIndex(Number(el.pageJump.value) - 1));
  el.pageJump.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    selectImageByVisibleIndex(Number(el.pageJump.value) - 1);
  });
  document.querySelector("#saveManualBtn").addEventListener("click", saveManualFinal);
  document.querySelector("#approveSourceBtn").addEventListener("click", approveCurrentSource);
  document.querySelector("#repairBtn").addEventListener("click", runRepair);
  el.runExperimentsBtn.addEventListener("click", runExperiments);
  document.querySelectorAll(".model-form").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      runOcr(form.dataset.provider, form);
    });
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setActiveSource(tab.dataset.source));
  });
  el.sourceImage.addEventListener("load", positionCropBox);
  window.addEventListener("resize", positionCropBox);
}

function renderConfig() {
  let gpt = "GPT OK";
  if (!state.config.gpt_oauth.configured) {
    gpt = "GPT key 없음";
  } else if (state.config.gpt_oauth.image_warning) {
    gpt = "GPT 이미지 URL 필요";
  }
  const chandra = state.config.chandra.configured ? "Chandra OK" : "Chandra 설정 없음";
  el.configStatus.textContent = `${chandra} · ${gpt}`;
  if (state.config.gpt_oauth.image_warning) {
    document.querySelector("#gptRunBtn").disabled = true;
    document.querySelector("#job-gpt").textContent = state.config.gpt_oauth.image_warning;
    document.querySelector("#job-gpt").classList.add("error");
  }
  if (!state.config.gpt_oauth.configured) {
    const btn = document.querySelector("#gptRunBtn");
    btn.disabled = true;
    document.querySelector("#job-gpt").textContent = "DOGOK_PROXY_API_KEY 필요";
    document.querySelector("#job-gpt").classList.add("error");
  }
  document.querySelector('form[data-provider="chandra"] input[name="model"]').value = state.config.chandra.default_model;
  populateSelectOptions(
    document.querySelector('form[data-provider="gpt"] select[name="model"]'),
    state.config.gpt_oauth.models || [state.config.gpt_oauth.default_model],
    state.config.gpt_oauth.default_model,
  );
  populateSelectOptions(
    document.querySelector('form[data-provider="gpt"] select[name="reasoning_effort"]'),
    state.config.gpt_oauth.reasoning_efforts || ["low", "medium", "high", "xhigh", "max"],
    "low",
  );
}

function populateSelectOptions(select, values, selectedValue) {
  if (!select) return;
  const currentValue = select.value || selectedValue;
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === currentValue || (!values.includes(currentValue) && value === selectedValue);
    select.appendChild(option);
  });
}

async function loadFolder(folderId, options = {}) {
  const selectFirst = options.selectFirst !== false;
  if (!folderId) {
    state.images = [];
    state.runs = [];
    state.selectedImage = null;
    state.currentRun = null;
    renderFolders();
    renderImages();
    renderRuns();
    renderEmptyGrid();
    setRunStatus("이미지 폴더 없음", true);
    return;
  }
  state.selectedFolderId = folderId;
  setSelectedFolderUrl(folderId);
  state.selectedImage = null;
  state.currentRun = null;
  state.activeSource = "chandra";
  state.imageFilter = "";
  el.imageSearch.value = "";
  renderFolders();
  setRunStatus("폴더 이미지 불러오는 중");
  const data = await apiGet(`/api/images?folder_id=${encodeURIComponent(folderId)}`);
  state.images = data.images || [];
  renderImages();
  await refreshRuns();
  if (selectFirst && state.images.length > 0) {
    await selectImage(state.images[0].id);
  } else {
    renderEmptyGrid();
    setRunStatus(state.images.length ? "run 불러오는 중" : "선택한 폴더에 이미지 없음", !state.images.length);
  }
}

function renderFolders() {
  el.folderList.innerHTML = "";
  el.folderSelect.innerHTML = "";
  el.folderSummary.textContent = `${state.folders.length}개`;
  if (!state.folders.length) {
    const option = document.createElement("option");
    option.textContent = "폴더 없음";
    option.value = "";
    el.folderSelect.appendChild(option);
    el.folderSelect.disabled = true;
    const empty = document.createElement("p");
    empty.className = "empty-line";
    empty.textContent = "폴더 없음";
    el.folderList.appendChild(empty);
    return;
  }
  el.folderSelect.disabled = false;
  state.folders.forEach((folder) => {
    const option = document.createElement("option");
    option.value = folder.id;
    option.textContent = `${folder.name} (${folder.image_count}장)`;
    el.folderSelect.appendChild(option);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "folder-item";
    button.dataset.id = folder.id;
    button.title = folder.relative_path;
    button.innerHTML = `<span class="folder-name">${escapeHtml(folder.name)}</span><small>${folder.image_count}장 · ${escapeHtml(folder.relative_path)}</small>`;
    button.classList.toggle("active", folder.id === state.selectedFolderId);
    button.addEventListener("click", () => loadFolder(folder.id));
    el.folderList.appendChild(button);
  });
  el.folderSelect.value = state.selectedFolderId || "";
}

function renderImages() {
  el.imageList.innerHTML = "";
  const images = filteredImages();
  const visibleEntries = visibleImageEntries(images);
  const lastVisible = visibleEntries[visibleEntries.length - 1];
  const rangeText = visibleEntries.length ? ` · 표시 ${visibleEntries[0].index + 1}-${lastVisible.index + 1}` : "";
  el.imageSummary.textContent = state.imageFilter ? `${images.length}/${state.images.length}장${rangeText}` : `${state.images.length}장${rangeText}`;
  if (!images.length) {
    const empty = document.createElement("p");
    empty.className = "empty-line";
    empty.textContent = state.images.length ? "검색 결과 없음" : "이미지 없음";
    el.imageList.appendChild(empty);
    updateImageNavigator();
    return;
  }
  visibleEntries.forEach(({ image, index }) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "image-item";
    button.dataset.id = image.id;
    button.title = image.relative_path || image.name;
    button.innerHTML = `<span class="image-main">${escapeHtml(image.name)}</span><small>${index + 1}/${images.length} · ${image.width}x${image.height}</small>`;
    button.classList.toggle("active", state.selectedImage?.id === image.id);
    button.addEventListener("click", () => selectImage(image.id));
    el.imageList.appendChild(button);
  });
  updateImageNavigator();
}

function visibleImageEntries(images) {
  const maxVisible = 9;
  if (images.length <= maxVisible) {
    return images.map((image, index) => ({ image, index }));
  }
  const current = currentVisibleImageIndex();
  const center = current >= 0 ? current : 0;
  const half = Math.floor(maxVisible / 2);
  let start = Math.max(0, center - half);
  let end = Math.min(images.length, start + maxVisible);
  start = Math.max(0, end - maxVisible);
  return images.slice(start, end).map((image, offset) => ({ image, index: start + offset }));
}

function setSelectedFolderUrl(folderId) {
  const url = new URL(window.location.href);
  if (url.searchParams.get("folder_id") === folderId) return;
  url.searchParams.set("folder_id", folderId);
  window.history.replaceState(null, "", url);
}

function setRunUrl(runId, folderId) {
  const url = new URL(window.location.href);
  if (folderId) url.searchParams.set("folder_id", folderId);
  url.searchParams.set("run_id", runId);
  window.history.replaceState(null, "", url);
}

function filteredImages() {
  if (!state.imageFilter) return state.images;
  return state.images.filter((image) => {
    const haystack = `${image.name} ${image.relative_path || ""}`.toLowerCase();
    return haystack.includes(state.imageFilter);
  });
}

async function selectAdjacentImage(offset) {
  const images = filteredImages();
  if (!images.length) return;
  const current = currentVisibleImageIndex();
  const base = current >= 0 ? current : 0;
  const next = Math.min(Math.max(base + offset, 0), images.length - 1);
  await selectImage(images[next].id);
}

async function selectImageByVisibleIndex(index) {
  const images = filteredImages();
  if (!images.length || Number.isNaN(index)) {
    updateImageNavigator();
    return;
  }
  const next = Math.min(Math.max(index, 0), images.length - 1);
  await selectImage(images[next].id);
}

function currentVisibleImageIndex() {
  if (!state.selectedImage) return -1;
  return filteredImages().findIndex((image) => image.id === state.selectedImage.id);
}

function updateImageNavigator() {
  const images = filteredImages();
  const index = currentVisibleImageIndex();
  const hasImages = images.length > 0;
  el.pageJump.disabled = !hasImages;
  el.prevImageBtn.disabled = !hasImages || index <= 0;
  el.nextImageBtn.disabled = !hasImages || index < 0 || index >= images.length - 1;
  el.pageJump.min = "1";
  el.pageJump.max = String(Math.max(images.length, 1));
  el.pageJump.value = index >= 0 ? String(index + 1) : "";
  el.imagePager.textContent = `${index >= 0 ? index + 1 : 0} / ${images.length}`;
}

async function refreshRuns() {
  const query = state.selectedFolderId ? `?folder_id=${encodeURIComponent(state.selectedFolderId)}` : "";
  const [runData, jobData] = await Promise.all([
    apiGet(`/api/runs${query}`),
    apiGet(`/api/jobs${query}`),
  ]);
  state.runs = runData.runs || [];
  syncTrackedJobs(jobData.jobs || []);
  renderRuns();
  syncJobLinesFromRun();
}

async function refreshAuditReports() {
  const data = await apiGet("/api/audit-reports");
  state.auditReports = data.reports || [];
  renderAuditReports();
}

function renderRuns() {
  el.runList.innerHTML = "";
  if (!state.runs.length) {
    const empty = document.createElement("p");
    empty.className = "empty-line";
    empty.textContent = "저장된 run 없음";
    el.runList.appendChild(empty);
    return;
  }
  state.runs.forEach((run) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-item";
    button.dataset.id = run.run_id;
    button.innerHTML = `<span>${escapeHtml(run.image_name || run.run_id)}</span><small>${run.run_id} · ${formatRunBadges(run)}</small>`;
    button.classList.toggle("active", state.currentRun?.run_id === run.run_id);
    button.addEventListener("click", () => loadRun(run.run_id));
    el.runList.appendChild(button);
  });
}

function renderAuditReports() {
  el.auditReportList.innerHTML = "";
  if (!state.auditReports.length) {
    const empty = document.createElement("p");
    empty.className = "empty-line";
    empty.textContent = "생성된 리포트 없음";
    el.auditReportList.appendChild(empty);
    return;
  }
  state.auditReports.forEach((report) => {
    const card = document.createElement("section");
    card.className = "audit-report-card";
    const summary = report.summary || {};
    const links = [
      report.ocr_results_url ? `<a href="${escapeHtml(report.ocr_results_url)}" target="_blank">OCR</a>` : "",
      report.report_url ? `<a href="${escapeHtml(report.report_url)}" target="_blank">Audit</a>` : "",
      report.contact_sheet_url ? `<a href="${escapeHtml(report.contact_sheet_url)}" target="_blank">Sheet</a>` : "",
    ].filter(Boolean).join("");
    card.innerHTML = `
      <div class="audit-report-head">
        <strong>${escapeHtml(report.id)}</strong>
        <span>${escapeHtml(String(report.count || 0))}개 · OK ${escapeHtml(String(summary.ok ?? "-"))}</span>
      </div>
      <div class="audit-report-links">${links}</div>
      <div class="audit-run-list"></div>
    `;
    const runList = card.querySelector(".audit-run-list");
    (report.items || []).forEach((item) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "audit-run-item";
      row.disabled = !item.run_id;
      row.innerHTML = `<span>${escapeHtml(item.index)}. ${escapeHtml(item.image_name || "")}</span><small>${escapeHtml(item.run_id || "run 없음")} · ${(item.flags || []).map(escapeHtml).join(", ")}</small>`;
      row.addEventListener("click", () => loadRun(item.run_id));
      runList.appendChild(row);
    });
    el.auditReportList.appendChild(card);
  });
}

function formatRunBadges(run) {
  const badges = [];
  if (run.has_chandra) badges.push("Chandra");
  if (run.has_gpt) badges.push("GPT");
  if (run.has_repair) badges.push("Repair");
  if (run.has_experiments) badges.push("Experiments");
  if (run.has_final) badges.push("Final");
  trackedJobsForRun(run.run_id).forEach((job) => {
    const label = job.kind === "repair" ? "Repair" : job.kind === "experiment" ? "Experiment" : (job.payload?.provider || "job").toUpperCase();
    const percent = Math.round((job.progress || 0) * 100);
    badges.push(`${label} ${job.status} ${percent}%`);
  });
  return badges.length ? badges.join("/") : "crop";
}

function syncTrackedJobs(jobs) {
  const activeJobs = {};
  jobs.forEach((job) => {
    if (!["queued", "running"].includes(job.status)) return;
    activeJobs[job.id] = job;
    ensureJobPolling(job);
  });
  Object.keys(state.jobs).forEach((jobId) => {
    if (activeJobs[jobId] || state.jobs[jobId]?.status === "queued" || state.jobs[jobId]?.status === "running") return;
    delete state.jobs[jobId];
  });
  state.jobs = { ...state.jobs, ...activeJobs };
}

function ensureJobPolling(job) {
  if (state.pollers[job.id]) return;
  const runId = job.payload?.run_id;
  const source = job.kind === "repair" ? "repair" : job.kind === "experiment" ? "experiment" : job.payload?.provider;
  pollJob(job.id, jobLineSelectorForSource(source), async () => {
    await refreshRuns();
    if (state.currentRun?.run_id === runId) {
      state.currentRun = await apiGet(`/api/runs/${runId}`);
      renderRun();
      if (source && source !== "experiment") setActiveSource(source);
    }
  }, { runId, source });
}

function trackedJobsForRun(runId) {
  return Object.values(state.jobs).filter((job) => {
    if (!["queued", "running"].includes(job.status)) return false;
    return job.payload?.run_id === runId;
  });
}

function trackedJobForSource(runId, source) {
  return trackedJobsForRun(runId).find((job) => {
    const jobSource = job.kind === "repair" ? "repair" : job.kind === "experiment" ? "experiment" : job.payload?.provider;
    return jobSource === source;
  });
}

function jobLineSelectorForSource(source) {
  if (source === "repair") return "#job-repair";
  if (source === "experiment") return "#job-experiment";
  return `#job-${source}`;
}

async function loadRun(runId) {
  try {
    state.currentRun = await apiGet(`/api/runs/${runId}`);
    if (state.currentRun.folder_id && state.currentRun.folder_id !== state.selectedFolderId) {
      state.selectedFolderId = state.currentRun.folder_id;
      setSelectedFolderUrl(state.selectedFolderId);
      const data = await apiGet(`/api/images?folder_id=${encodeURIComponent(state.selectedFolderId)}`);
      state.images = data.images || [];
      renderFolders();
      renderImages();
      await refreshRuns();
    }
    state.selectedImage = state.images.find((image) => image.id === state.currentRun.image_id) || null;
    renderImages();
    document.querySelectorAll(".run-item").forEach((item) => {
      item.classList.toggle("active", item.dataset.id === runId);
    });
    updateImageNavigator();
    if (state.selectedImage) {
      el.imageTitle.textContent = state.selectedImage.name;
      el.sourceImage.src = state.selectedImage.url;
    } else {
      el.imageTitle.textContent = state.currentRun.image_name || runId;
    }
    fillBboxForm(state.currentRun.bbox);
    state.activeSource = bestSourceForRun(state.currentRun);
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.source === state.activeSource);
    });
    renderRun();
    setRunUrl(runId, state.currentRun.folder_id);
    setRunStatus(`run ${runId}`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

function bestSourceForRun(run) {
  if (run.repair) return "repair";
  if (run.ocr?.gpt) return "gpt";
  if (run.ocr?.chandra) return "chandra";
  return "chandra";
}

async function selectImage(imageId) {
  state.selectedImage = state.images.find((image) => image.id === imageId);
  if (!state.selectedImage) return;
  state.currentRun = null;
  state.activeSource = "chandra";
  renderImages();
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.source === "chandra");
  });
  el.imageTitle.textContent = state.selectedImage.name;
  el.runStatus.textContent = `${state.selectedImage.width} x ${state.selectedImage.height}`;
  el.sourceImage.src = state.selectedImage.url;
  el.cropPreview.removeAttribute("src");
  el.maskedPreview.removeAttribute("src");
  renderTrace(null, "chandra");
  el.experimentResults.innerHTML = `<p class="empty-line">아직 실험 결과 없음</p>`;
  el.finalLinks.innerHTML = "";
  renderEmptyGrid();
  const existingRun = latestRunForImage(imageId) || await fetchLatestRunForImage(imageId);
  if (existingRun) {
    await loadRun(existingRun.run_id);
    return;
  }
  await cropSelected(null);
}

function latestRunForImage(imageId) {
  return state.runs.find((run) => run.image_id === imageId) || null;
}

async function fetchLatestRunForImage(imageId) {
  const params = new URLSearchParams({ image_id: imageId });
  if (state.selectedFolderId) params.set("folder_id", state.selectedFolderId);
  const data = await apiGet(`/api/runs?${params.toString()}`);
  const run = data.runs?.[0] || null;
  if (run && !state.runs.some((item) => item.run_id === run.run_id)) {
    state.runs = [run, ...state.runs].sort((a, b) => (b.created_at || 0) - (a.created_at || 0));
    renderRuns();
  }
  return run;
}

async function cropSelected(bbox) {
  if (!state.selectedImage) return;
  setRunStatus("crop 생성 중");
  const payload = { image_id: state.selectedImage.id };
  if (bbox) payload.bbox = bbox;
  try {
    state.currentRun = await apiPost("/api/crop", payload);
    await refreshRuns();
    fillBboxForm(state.currentRun.bbox);
    renderRun();
    setRunStatus(`run ${state.currentRun.run_id}`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

function fillBboxForm(bbox) {
  el.bboxX.value = bbox.x;
  el.bboxY.value = bbox.y;
  el.bboxW.value = bbox.width;
  el.bboxH.value = bbox.height;
  positionCropBox();
}

function readBboxForm() {
  return {
    x: Number(el.bboxX.value || 0),
    y: Number(el.bboxY.value || 0),
    width: Number(el.bboxW.value || 1),
    height: Number(el.bboxH.value || 1),
  };
}

function positionCropBox() {
  if (!state.currentRun || !state.selectedImage || !el.sourceImage.complete) return;
  const bbox = state.currentRun.bbox;
  const rect = el.sourceImage.getBoundingClientRect();
  const parentRect = el.sourceImage.parentElement.getBoundingClientRect();
  const scaleX = rect.width / state.selectedImage.width;
  const scaleY = rect.height / state.selectedImage.height;
  el.cropBox.style.display = "block";
  el.cropBox.style.left = `${rect.left - parentRect.left + bbox.x * scaleX}px`;
  el.cropBox.style.top = `${rect.top - parentRect.top + bbox.y * scaleY}px`;
  el.cropBox.style.width = `${bbox.width * scaleX}px`;
  el.cropBox.style.height = `${bbox.height * scaleY}px`;
}

function renderRun() {
  if (!state.currentRun) return;
  const stamp = `?t=${Date.now()}`;
  el.cropPreview.src = state.currentRun.crop_url + stamp;
  el.maskedPreview.src = state.currentRun.masked_url + stamp;
  positionCropBox();
  syncJobLinesFromRun();
  renderSource(state.activeSource);
  renderExperiments();
  renderFinalLinks();
}

function syncJobLinesFromRun() {
  syncJobLine("#job-chandra", "chandra", Boolean(state.currentRun?.ocr?.chandra));
  syncJobLine("#job-gpt", "gpt", Boolean(state.currentRun?.ocr?.gpt));
  syncJobLine("#job-repair", "repair", Boolean(state.currentRun?.repair));
  syncJobLine("#job-experiment", "experiment", Boolean(Object.keys(state.currentRun?.experiments || {}).length));
  renderExperimentJobState();
}

function syncJobLine(selector, source, hasResult) {
  const line = document.querySelector(selector);
  if (!line) return;
  const runningJob = state.currentRun ? trackedJobForSource(state.currentRun.run_id, source) : null;
  if (runningJob) {
    line.textContent = formatJobStatus(runningJob);
    line.classList.remove("error");
    return;
  }
  if (hasResult) {
    line.textContent = "완료";
    line.classList.remove("error");
    return;
  }
  if (line.textContent.includes("running") || line.textContent.includes("요청 생성 중")) return;
  line.textContent = "대기";
  line.classList.remove("error");
}

async function refreshRun() {
  if (!state.currentRun) return;
  state.currentRun = await apiGet(`/api/runs/${state.currentRun.run_id}`);
  renderRun();
}

function collectSettings(form) {
  const data = new FormData(form);
  return {
    model: String(data.get("model") || "").trim(),
    reasoning_effort: String(data.get("reasoning_effort") || "low").trim(),
    temperature: Number(data.get("temperature") || 0.3),
    top_p: Number(data.get("top_p") || 0.95),
    max_tokens: Number(data.get("max_tokens") || 4096),
    n: Number(data.get("n") || 1),
  };
}

async function runOcr(provider, form) {
  if (!state.currentRun) return;
  const targetRunId = state.currentRun.run_id;
  const userConditions = currentUserConditions();
  const line = document.querySelector(`#job-${provider}`);
  line.classList.remove("error");
  line.textContent = "요청 생성 중";
  try {
    const job = await apiPost("/api/ocr", {
      run_id: targetRunId,
      provider,
      settings: collectSettings(form),
      user_conditions: userConditions,
    });
    state.jobs[job.id] = job;
    renderRuns();
    pollJob(job.id, `#job-${provider}`, async () => {
      await refreshRuns();
      const completedRun = await apiGet(`/api/runs/${targetRunId}`);
      if (state.currentRun?.run_id === targetRunId) {
        state.currentRun = completedRun;
        renderRun();
        setActiveSource(provider);
      }
      if (provider === "gpt") {
        await startRepairForRun({
          targetRunId,
          sourceProvider: "gpt",
          provider: "gpt",
          settings: collectSettings(form),
          userConditions,
          activateOnComplete: true,
        });
      }
    }, { runId: targetRunId, source: provider });
  } catch (error) {
    line.textContent = error.message;
    line.classList.add("error");
  }
}

async function runRepair() {
  if (!state.currentRun) return;
  const source = document.querySelector("#repairSource").value;
  const provider = document.querySelector("#repairProvider").value;
  const sourceForm = document.querySelector(`.model-form[data-provider="${provider}"]`);
  await startRepairForRun({
    targetRunId: state.currentRun.run_id,
    sourceProvider: source,
    provider,
    settings: collectSettings(sourceForm),
    userConditions: currentUserConditions(),
    activateOnComplete: true,
  });
}

async function runExperiments() {
  if (!state.currentRun) return;
  const provider = el.experimentProvider.value;
  const providerForm = document.querySelector(`.model-form[data-provider="${provider}"]`);
  const strategies = selectedExperimentStrategies();
  const line = document.querySelector("#job-experiment");
  line.classList.remove("error");
  if (!providerCanRun(provider)) {
    line.textContent = providerStatusMessage(provider);
    line.classList.add("error");
    return;
  }
  if (!strategies.length) {
    line.textContent = "전략을 하나 이상 선택하세요";
    line.classList.add("error");
    return;
  }
  line.textContent = "요청 생성 중";
  const targetRunId = state.currentRun.run_id;
  const pendingJob = { status: "queued", progress: 0, payload: { run_id: targetRunId, provider, strategies } };
  setExperimentButtonState(pendingJob);
  setExperimentPanelState(pendingJob);
  try {
    const job = await apiPost("/api/experiments", {
      run_id: targetRunId,
      provider,
      strategies,
      settings: collectSettings(providerForm),
      user_conditions: currentUserConditions(),
    });
    state.jobs[job.id] = job;
    line.textContent = formatJobStatus(job);
    renderExperimentJobState(job);
    renderRuns();
    pollJob(job.id, "#job-experiment", async () => {
      await refreshRuns();
      if (state.currentRun?.run_id === targetRunId) {
        state.currentRun = await apiGet(`/api/runs/${targetRunId}`);
        renderRun();
      }
    }, { runId: targetRunId, source: "experiment" });
  } catch (error) {
    line.textContent = error.message;
    line.classList.add("error");
    renderExperimentJobState({ status: "failed", progress: 1, error: error.message, payload: { run_id: targetRunId, provider, strategies } });
  }
}

function selectedExperimentStrategies() {
  return Array.from(document.querySelectorAll('input[name="experimentStrategy"]:checked')).map((input) => input.value);
}

function providerCanRun(provider) {
  if (provider === "gpt") {
    return Boolean(state.config?.gpt_oauth?.configured) && !state.config?.gpt_oauth?.image_warning;
  }
  if (provider === "chandra") {
    return Boolean(state.config?.chandra?.configured);
  }
  return false;
}

function providerStatusMessage(provider) {
  if (provider === "gpt") {
    if (!state.config?.gpt_oauth?.configured) return "DOGOK_PROXY_API_KEY 필요";
    return state.config?.gpt_oauth?.image_warning || "GPT 설정 확인 필요";
  }
  if (provider === "chandra") return "Chandra 설정 확인 필요";
  return "모델 설정 확인 필요";
}

async function startRepairForRun({ targetRunId, sourceProvider, provider, settings, userConditions, activateOnComplete }) {
  const line = document.querySelector("#job-repair");
  const isVisibleTarget = () => state.currentRun?.run_id === targetRunId;
  if (isVisibleTarget()) {
    line.classList.remove("error");
    line.textContent = "요청 생성 중";
  }
  try {
    const currentPayload = state.currentRun?.run_id === targetRunId ? state.currentRun : await apiGet(`/api/runs/${targetRunId}`);
    if (currentPayload.repair) {
      if (isVisibleTarget()) {
        line.textContent = "완료";
        state.currentRun = currentPayload;
        renderRun();
        if (activateOnComplete) setActiveSource("repair");
      }
      return;
    }
    const existingRepairJob = trackedJobForSource(targetRunId, "repair");
    if (existingRepairJob) {
      if (isVisibleTarget()) line.textContent = formatJobStatus(existingRepairJob);
      return;
    }
    const job = await apiPost("/api/repair", {
      run_id: targetRunId,
      source_provider: sourceProvider,
      provider,
      settings,
      user_conditions: userConditions,
    });
    state.jobs[job.id] = job;
    renderRuns();
    pollJob(job.id, "#job-repair", async () => {
      await refreshRuns();
      if (state.currentRun?.run_id === targetRunId) {
        state.currentRun = await apiGet(`/api/runs/${targetRunId}`);
        renderRun();
        if (activateOnComplete) setActiveSource("repair");
      }
    }, { runId: targetRunId, source: "repair" });
  } catch (error) {
    if (isVisibleTarget()) {
      line.textContent = error.message;
      line.classList.add("error");
    }
  }
}

function currentUserConditions() {
  return el.userConditions?.value?.trim() || "";
}

async function pollJob(jobId, selector, onComplete, meta = {}) {
  if (state.pollers[jobId]) return;
  const line = document.querySelector(selector);
  const pollOnce = async () => {
    try {
      const job = await apiGet(`/api/jobs/${jobId}`);
      state.jobs[jobId] = job;
      if (line && (!meta.runId || state.currentRun?.run_id === meta.runId)) {
        line.textContent = formatJobStatus(job);
        line.classList.toggle("error", job.status === "failed");
      }
      if (meta.source === "experiment") renderExperimentJobState(job);
      renderRuns();
      if (job.status === "completed") {
        window.clearInterval(timer);
        delete state.pollers[jobId];
        delete state.jobs[jobId];
        if (line && (!meta.runId || state.currentRun?.run_id === meta.runId)) {
          line.textContent = "완료";
          line.classList.remove("error");
        }
        await onComplete();
        if (meta.source === "experiment") renderExperimentJobState();
      }
      if (job.status === "failed") {
        window.clearInterval(timer);
        delete state.pollers[jobId];
        delete state.jobs[jobId];
        if (line && (!meta.runId || state.currentRun?.run_id === meta.runId)) {
          line.textContent = job.error || "실패";
          line.classList.add("error");
        }
        await refreshRuns();
        if (meta.source === "experiment") renderExperimentJobState(job);
      }
    } catch (error) {
      window.clearInterval(timer);
      delete state.pollers[jobId];
      delete state.jobs[jobId];
      if (line && (!meta.runId || state.currentRun?.run_id === meta.runId)) {
        line.textContent = error.message;
        line.classList.add("error");
      }
      await refreshRuns();
      if (meta.source === "experiment") renderExperimentJobState({ status: "failed", progress: 1, error: error.message, payload: { run_id: meta.runId } });
    }
  };
  const timer = window.setInterval(pollOnce, 1500);
  state.pollers[jobId] = timer;
  pollOnce();
}

function formatJobStatus(job) {
  const percent = Math.round((job.progress || 0) * 100);
  const remoteStatus = job.remote_status ? ` · dogok ${job.remote_status}` : "";
  const strategy = job.active_strategy ? ` · ${strategyLabel(job.active_strategy)}` : "";
  const chunk = formatJobChunk(job);
  return `${job.status} ${percent}%${remoteStatus}${strategy}${chunk}`;
}

function formatJobChunk(job) {
  const chunk = job.active_chunk || {};
  if (!chunk.row_count) return "";
  const rowStart = Number(chunk.row_start || 0) + 1;
  const rowEnd = rowStart + Number(chunk.row_count || 1) - 1;
  const chunkIndex = chunk.index && chunk.total ? ` ${chunk.index}/${chunk.total}` : "";
  return ` · row ${rowStart}-${rowEnd}${chunkIndex}`;
}

function renderExperimentJobState(job = null) {
  const currentJob = job && (!state.currentRun || job.payload?.run_id === state.currentRun.run_id)
    ? job
    : state.currentRun
      ? trackedJobForSource(state.currentRun.run_id, "experiment")
      : null;
  const experiments = state.currentRun?.experiments || {};
  const providers = Object.keys(experiments);
  const hasResult = Boolean(providers.length);
  if (currentJob) {
    setExperimentButtonState(currentJob);
    setExperimentPanelState(currentJob);
    return;
  }
  if (hasResult) {
    setExperimentButtonState(null);
    setExperimentPanelState({
      status: "completed",
      progress: 1,
      payload: {
        run_id: state.currentRun?.run_id,
        provider: providers.join(", "),
        strategies: completedExperimentStrategies(experiments),
      },
    });
    return;
  }
  setExperimentButtonState(null);
  setExperimentPanelState(null);
}

function setExperimentButtonState(job) {
  if (!el.runExperimentsBtn) return;
  const isRunning = job && ["queued", "running"].includes(job.status);
  el.runExperimentsBtn.disabled = Boolean(isRunning);
  if (isRunning) {
    const percent = Math.round((job.progress || 0) * 100);
    el.runExperimentsBtn.textContent = `실험 실행 중 ${percent}%`;
  } else {
    el.runExperimentsBtn.textContent = "선택 실험 실행";
  }
}

function setExperimentPanelState(job) {
  if (!el.experimentJobPanel) return;
  const percent = Math.round(((job?.progress || 0) * 100));
  const status = job?.status || "idle";
  const isFailed = status === "failed";
  el.experimentJobPanel.classList.toggle("is-idle", !job);
  el.experimentJobPanel.classList.toggle("is-running", ["queued", "running"].includes(status));
  el.experimentJobPanel.classList.toggle("is-failed", isFailed);
  el.experimentJobState.textContent = isFailed ? (job.error || "실패") : statusLabel(status);
  el.experimentJobTarget.textContent = formatExperimentJobTarget(job);
  el.experimentProgressFill.style.width = `${percent}%`;
  el.experimentProgressText.textContent = `${percent}%`;
  el.experimentJobStrategy.textContent = formatExperimentJobStrategy(job);
}

function statusLabel(status) {
  const labels = {
    idle: "대기",
    queued: "요청 생성 중",
    running: "실행 중",
    completed: "완료",
    failed: "실패",
  };
  return labels[status] || status;
}

function formatExperimentJobTarget(job) {
  if (!job) return "-";
  const provider = job.payload?.provider || el.experimentProvider?.value || "-";
  const runId = job.payload?.run_id || state.currentRun?.run_id || "-";
  return `${provider} · ${runId}`;
}

function formatExperimentJobStrategy(job) {
  if (!job) return "-";
  const chunk = formatJobChunk(job).replace(/^ · /, "");
  const active = job.active_strategy ? strategyLabel(job.active_strategy) : "";
  const strategies = (job.payload?.strategies || []).map(strategyLabel).join(", ");
  if (active && chunk) return `${active} · ${chunk}`;
  if (active) return active;
  return strategies || "-";
}

function completedExperimentStrategies(experiments) {
  const strategies = [];
  Object.values(experiments || {}).forEach((summary) => {
    (summary.strategies || []).forEach((strategy) => {
      if (!strategies.includes(strategy)) strategies.push(strategy);
    });
    (summary.variants || []).forEach((variant) => {
      if (variant.strategy && !strategies.includes(variant.strategy)) strategies.push(variant.strategy);
    });
  });
  return strategies;
}

function setActiveSource(source) {
  state.activeSource = source;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.source === source);
  });
  renderSource(source);
}

function renderSource(source) {
  const result = getSourceResult(source);
  if (!result) {
    renderEmptyGrid();
    renderTrace(null, source);
    el.validationStatus.textContent = `${source} 결과 없음`;
    return;
  }
  const cells = result.cells || result.rows || [];
  renderCells(cells);
  renderPaperPreview(cells);
  renderTrace(result, source);
  renderValidation(result.validation);
}

function renderExperiments() {
  el.experimentResults.innerHTML = "";
  const experiments = state.currentRun?.experiments || {};
  const providers = Object.keys(experiments);
  if (!providers.length) {
    const empty = document.createElement("p");
    empty.className = "empty-line";
    empty.textContent = "아직 실험 결과 없음";
    el.experimentResults.appendChild(empty);
    return;
  }
  providers.forEach((provider) => {
    const group = document.createElement("section");
    group.className = "experiment-group";
    const summary = experiments[provider] || {};
    const variants = summary.variants || [];
    group.innerHTML = `<h3>${escapeHtml(provider)} · ${variants.length}개 전략</h3>`;
    const list = document.createElement("div");
    list.className = "experiment-card-list";
    variants.forEach((variant) => {
      const card = document.createElement("article");
      card.className = "experiment-card";
      const voteStats = variant.validation?.vote_stats || {};
      const metric = variant.strategy === "vote_full_row_1_2"
        ? `disagree ${voteStats.disagreement_cells ?? "-"} · majority ${voteStats.majority_cells ?? "-"}`
        : `shape ${variant.validation?.normalized_shape?.join("x") || "?"}`;
      card.innerHTML = `
        <div>
          <strong>${escapeHtml(strategyLabel(variant.strategy))}</strong>
          <small>${escapeHtml(metric)}</small>
        </div>
        <div class="experiment-actions">
          <button type="button" data-provider="${escapeHtml(provider)}" data-strategy="${escapeHtml(variant.strategy)}">결과 보기</button>
          ${variant.render_url ? `<a href="${escapeHtml(variant.render_url)}" target="_blank">HTML</a>` : ""}
        </div>
      `;
      card.querySelector("button").addEventListener("click", () => renderExperimentResult(provider, variant.strategy));
      list.appendChild(card);
    });
    group.appendChild(list);
    el.experimentResults.appendChild(group);
  });
}

function strategyLabel(strategy) {
  const labels = {
    full_grid: "전체 crop",
    row_1: "1줄씩",
    row_2: "2줄씩",
    row_5: "5줄씩",
    vote_full_row_1_2: "Voting",
  };
  return labels[strategy] || strategy;
}

function renderExperimentResult(provider, strategy) {
  const variant = experimentVariant(provider, strategy);
  if (!variant) return;
  state.activeExperiment = { provider, strategy };
  renderCells(variant.cells || variant.rows || []);
  renderPaperPreview(variant.cells || variant.rows || []);
  renderTrace(variant, `experiment:${strategy}`);
  renderValidation(variant.validation);
  setRunStatus(`실험 결과 보기 · ${provider} · ${strategyLabel(strategy)}`);
}

function experimentVariant(provider, strategy) {
  const variants = state.currentRun?.experiments?.[provider]?.variants || [];
  return variants.find((variant) => variant.strategy === strategy) || null;
}

function renderTrace(result, source) {
  if (!result) {
    el.traceSummary.textContent = `${source} 결과 없음`;
    el.traceInput.textContent = "";
    el.tracePrompt.textContent = "";
    el.traceRequest.textContent = "";
    el.rawOutput.textContent = "";
    return;
  }
  const trace = result.trace || {};
  const provider = trace.provider || result.provider || source;
  const kind = trace.kind || (source === "repair" ? "repair" : "ocr");
  const model = result.settings?.model || trace.settings?.model || "";
  el.traceSummary.textContent = `${kind} · ${provider}${model ? ` · ${model}` : ""}`;
  el.traceInput.textContent = stringifyTrace(buildTraceInput(trace, result, source));
  el.tracePrompt.textContent = trace.prompt || "trace.prompt 없음";
  el.traceRequest.textContent = stringifyTrace(trace.request_shape || buildFallbackRequest(result, source));
  el.rawOutput.textContent = result.raw_output || JSON.stringify(result, null, 2);
}

function buildTraceInput(trace, result, source) {
  const input = trace.input || {};
  const payload = {
    source,
    provider: trace.provider || result.provider || source,
    settings: result.settings || trace.settings || {},
    image: input.image || null,
  };
  if (trace.strategy || result.strategy) payload.strategy = trace.strategy || result.strategy;
  if (input.chunks) payload.chunks = input.chunks;
  if (input.source_provider) payload.source_provider = input.source_provider;
  if (input.source_cells) payload.source_cells = input.source_cells;
  if (input.source_cells_json && !input.source_cells) payload.source_cells_json = input.source_cells_json;
  if (input.user_conditions) payload.user_conditions = input.user_conditions;
  return payload;
}

function buildFallbackRequest(result, source) {
  return {
    note: "이 결과는 trace 저장 전 생성되어 API가 재구성한 정보만 표시합니다.",
    source,
    provider: result.provider || source,
    settings: result.settings || {},
  };
}

function stringifyTrace(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value ?? {}, null, 2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function getSourceResult(source) {
  if (!state.currentRun) return null;
  if (source === "repair") return state.currentRun.repair || null;
  return state.currentRun.ocr?.[source] || null;
}

function renderEmptyGrid() {
  renderCells(emptyCells());
  renderPaperPreview(emptyCells());
}

function renderCells(cells) {
  el.matrixEditor.innerHTML = "";
  normalizeCells(cells).forEach((row, rowIndex) => {
    row.forEach((cellText, columnIndex) => {
      const input = document.createElement("input");
      input.value = cellText === " " ? "" : cellText;
      input.maxLength = 2;
      input.dataset.row = rowIndex;
      input.dataset.column = columnIndex;
      input.classList.toggle("multi-cell", [...input.value].length > 1);
      input.addEventListener("input", () => {
        input.value = normalizeCellValue(input.value) === " " ? "" : normalizeCellValue(input.value);
        input.classList.toggle("multi-cell", [...input.value].length > 1);
        renderPaperPreview(readEditorCells());
      });
      input.addEventListener("keydown", (event) => moveCellFocus(event, rowIndex, columnIndex));
      el.matrixEditor.appendChild(input);
    });
  });
}

function moveCellFocus(event, row, column) {
  const moves = {
    ArrowRight: [row, column + 1],
    ArrowLeft: [row, column - 1],
    ArrowDown: [row + 1, column],
    ArrowUp: [row - 1, column],
  };
  if (!moves[event.key]) return;
  const [nextRow, nextColumn] = moves[event.key];
  if (nextRow < 0 || nextRow > 19 || nextColumn < 0 || nextColumn > 19) return;
  event.preventDefault();
  const next = el.matrixEditor.querySelector(`input[data-row="${nextRow}"][data-column="${nextColumn}"]`);
  if (next) next.focus();
}

function readEditorCells() {
  const rows = Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => " "));
  el.matrixEditor.querySelectorAll("input").forEach((input) => {
    const row = Number(input.dataset.row);
    const column = Number(input.dataset.column);
    rows[row][column] = normalizeCellValue(input.value);
  });
  return rows;
}

function renderPaperPreview(cells) {
  el.htmlPreview.innerHTML = "";
  normalizeCells(cells).forEach((row) => {
    row.forEach((cellText) => {
      const cell = document.createElement("span");
      cell.className = "paper-cell";
      cell.classList.toggle("multi-cell", [...cellText.trim()].length > 1);
      cell.textContent = cellText;
      el.htmlPreview.appendChild(cell);
    });
  });
}

function normalizeCells(cells) {
  const normalized = [];
  for (let i = 0; i < 20; i += 1) {
    const sourceRow = Array.isArray(cells?.[i]) ? cells[i] : [...String(cells?.[i] || "")];
    const row = sourceRow.map((cellText) => normalizeCellValue(cellText));
    while (row.length < 20) row.push(" ");
    normalized.push(row.slice(0, 20));
  }
  return normalized;
}

function normalizeCellValue(value) {
  const text = String(value || "").trim();
  if (!text) return " ";
  return [...text].slice(0, 2).join("");
}

function emptyCells() {
  return Array.from({ length: 20 }, () => Array.from({ length: 20 }, () => " "));
}

function renderValidation(validation) {
  if (!validation) {
    el.validationStatus.textContent = "검증 정보 없음";
    return;
  }
  const ok = validation.valid_original_shape ? "원본 20x20" : "정규화됨";
  el.validationStatus.textContent = `${ok} · rows=${validation.row_count} · shape=${validation.normalized_shape.join("x")}`;
}

async function saveManualFinal() {
  if (!state.currentRun) return;
  try {
    state.currentRun = await apiPost("/api/finalize", {
      run_id: state.currentRun.run_id,
      source: "manual",
      cells: readEditorCells(),
    });
    renderRun();
    setRunStatus(`final 저장 ${state.currentRun.run_id}`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

async function approveCurrentSource() {
  if (!state.currentRun) return;
  if (!getSourceResult(state.activeSource)) {
    setRunStatus(`${state.activeSource} 결과 없음`, true);
    return;
  }
  try {
    state.currentRun = await apiPost("/api/finalize", {
      run_id: state.currentRun.run_id,
      source: state.activeSource,
    });
    renderRun();
    setRunStatus(`final 저장 ${state.currentRun.run_id}`);
  } catch (error) {
    setRunStatus(error.message, true);
  }
}

function renderFinalLinks() {
  el.finalLinks.innerHTML = "";
  if (!state.currentRun?.final) return;
  const htmlLink = document.createElement("a");
  htmlLink.href = state.currentRun.final.render_url;
  htmlLink.target = "_blank";
  htmlLink.textContent = "final.html";
  el.finalLinks.appendChild(htmlLink);
}

function setRunStatus(message, isError = false) {
  el.runStatus.textContent = message;
  el.runStatus.classList.toggle("error", isError);
}

async function apiGet(url) {
  const response = await fetch(url);
  return parseResponse(response);
}

async function apiPost(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail);
  }
  return data;
}

boot().catch((error) => {
  setRunStatus(error.message, true);
});
