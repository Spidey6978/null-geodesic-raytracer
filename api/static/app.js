/**
 * KERR-TRACE v1.0 — Minimal Scientific Software Interface Controller
 * Handles tab switching, slider syncing, real-time render job polling, and viewport updates.
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- DOM Elements ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");
    const paramDrawer = document.getElementById("param-drawer");

    const sliderSpin = document.getElementById("slider-spin");
    const valSpin = document.getElementById("val-spin");
    const sliderMass = document.getElementById("slider-mass");
    const valMass = document.getElementById("val-mass");
    const sliderInclination = document.getElementById("slider-inclination");
    const valInclination = document.getElementById("val-inclination");

    const selectPreset = document.getElementById("select-preset");
    const sliderFov = document.getElementById("slider-fov");
    const valFov = document.getElementById("val-fov");
    const selectQuality = document.getElementById("select-quality");

    const sliderDiskOuter = document.getElementById("slider-disk-outer");
    const valDiskOuter = document.getElementById("val-disk-outer");
    const selectSkybox = document.getElementById("select-skybox");

    const btnRenderHero = document.getElementById("btn-render-hero");
    const frameStatusText = document.getElementById("frame-completion-text");
    const renderViewportImg = document.getElementById("render-viewport-img");

    const statRenderTime = document.getElementById("stat-render-time");
    const statIsco = document.getElementById("stat-isco");
    const statEscape = document.getElementById("stat-escape");

    const btnDownloadPng = document.getElementById("btn-download-png");
    const btnCopyJson = document.getElementById("btn-copy-json");

    let currentJobId = null;
    let currentJobMeta = null;
    let pollInterval = null;

    // --- 1. Tab Switcher Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPanel = document.getElementById(`tab-panel-${targetTab}`);
            if (targetPanel) {
                targetPanel.classList.add("active");
            }
            paramDrawer.classList.add("drawer-expanded");
        });
    });

    // --- 2. Live Value Syncing (No Input Boxes) ---
    sliderSpin.addEventListener("input", (e) => {
        valSpin.textContent = parseFloat(e.target.value).toFixed(3);
    });

    sliderMass.addEventListener("input", (e) => {
        valMass.textContent = `${parseFloat(e.target.value).toFixed(1)} M☉`;
    });

    sliderInclination.addEventListener("input", (e) => {
        valInclination.textContent = `${e.target.value}°`;
    });

    sliderFov.addEventListener("input", (e) => {
        valFov.textContent = `${e.target.value}°`;
    });

    sliderDiskOuter.addEventListener("input", (e) => {
        valDiskOuter.textContent = `${e.target.value} r_s`;
    });

    selectPreset.addEventListener("change", (e) => {
        const preset = e.target.value;
        if (preset === "hero") {
            sliderInclination.value = 70;
            valInclination.textContent = "70°";
        } else if (preset === "luminet_1979") {
            sliderInclination.value = 85;
            valInclination.textContent = "85°";
        } else if (preset === "high_inclination") {
            sliderInclination.value = 88;
            valInclination.textContent = "88°";
        }
    });

    // --- 3. Job Submission & Real-Time Polling ---
    btnRenderHero.addEventListener("click", async () => {
        const spin = parseFloat(sliderSpin.value);
        const mass = parseFloat(sliderMass.value);
        const fov = parseFloat(sliderFov.value);
        const diskOuter = parseFloat(sliderDiskOuter.value);
        const qualityMode = selectQuality.value;
        const skyboxPath = selectSkybox.value === "procedural" ? "procedural" : null;

        const payload = {
            black_hole: {
                mass: mass,
                spin: spin,
                disk_outer: diskOuter
            },
            camera: {
                preset: selectPreset.value,
                fov: fov
            },
            mode: qualityMode,
            skybox_path: skyboxPath
        };

        btnRenderHero.disabled = true;
        btnRenderHero.style.opacity = "0.5";
        frameStatusText.textContent = "Frame 0/1 | Submitting job...";

        try {
            const resp = await fetch("/api/v1/renders/image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                throw new Error(`HTTP Error ${resp.status}`);
            }

            const data = await resp.json();
            currentJobId = data.job_id;
            frameStatusText.textContent = `Frame 0/1 (0%) | Status: ${data.status}`;

            // Start real-time job status polling
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(() => pollJobStatus(currentJobId), 600);

        } catch (err) {
            console.error("Failed to submit render job:", err);
            frameStatusText.textContent = `❌ Submission Error: ${err.message}`;
            btnRenderHero.disabled = false;
            btnRenderHero.style.opacity = "1.0";
        }
    });

    async function pollJobStatus(jobId) {
        try {
            const resp = await fetch(`/api/v1/jobs/${jobId}`);
            if (!resp.ok) return;

            const job = await resp.json();
            const pct = Math.round(job.progress_pct || 0);

            if (job.status === "COMPLETED") {
                clearInterval(pollInterval);
                frameStatusText.textContent = `Frame 1/1 (100%) | Render Complete`;
                btnRenderHero.disabled = false;
                btnRenderHero.style.opacity = "1.0";

                // Update Viewport Image
                const imgUrl = `/api/v1/jobs/${jobId}/image?t=${Date.now()}`;
                renderViewportImg.src = imgUrl;

                // Fetch & Display Metadata
                fetchJobMetadata(jobId);
            } else if (job.status === "FAILED") {
                clearInterval(pollInterval);
                frameStatusText.textContent = `❌ Job Failed: ${job.error_message || "Unknown error"}`;
                btnRenderHero.disabled = false;
                btnRenderHero.style.opacity = "1.0";
            } else {
                frameStatusText.textContent = `Frame 0/1 (${pct}%) | Status: ${job.status}`;
            }
        } catch (err) {
            console.warn("Polling status error:", err);
        }
    }

    async function fetchJobMetadata(jobId) {
        try {
            const resp = await fetch(`/api/v1/jobs/${jobId}/metadata`);
            if (!resp.ok) return;
            const meta = await resp.json();
            currentJobMeta = meta;

            if (statRenderTime) statRenderTime.textContent = `${meta.render_time_s.toFixed(2)}s`;
            if (statIsco) {
                const a = meta.spin;
                const z1 = 1.0 + Math.cbrt(1.0 - a*a) * (Math.cbrt(1.0 + a) + Math.cbrt(1.0 - a));
                const z2 = Math.sqrt(3.0 * a*a + z1*z1);
                const r_isco = 3.0 + z2 - Math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0*z2));
                statIsco.textContent = `${r_isco.toFixed(3)} M`;
            }
        } catch (err) {
            console.warn("Failed to fetch job metadata:", err);
        }
    }

    // --- 4. Export Actions ---
    btnDownloadPng.addEventListener("click", () => {
        if (!renderViewportImg.src) return;
        const a = document.createElement("a");
        a.href = renderViewportImg.src;
        a.download = `${currentJobId || 'kerr_blackhole'}.png`;
        a.click();
    });

    btnCopyJson.addEventListener("click", () => {
        if (!currentJobMeta) return;
        navigator.clipboard.writeText(JSON.stringify(currentJobMeta, null, 2));
        btnCopyJson.textContent = "✅ Copied!";
        setTimeout(() => { btnCopyJson.textContent = "📋 Copy Metadata JSON"; }, 2000);
    });
});
