const $ = (sel) => document.querySelector(sel);

let wsDashboard = null;
let streamRegistro = null;

let modoCapturaRegistro = "";
let coloresPersona = {};

const estado = { camaras: [], alertas: 0, firmaCamaras: "", principalId: null };

const MS_SIN_SENAL = 5000;

function estaSinSenal(cam) {
  return cam.senal_hace_ms == null || cam.senal_hace_ms > MS_SIN_SENAL;
}

function pedirCamara(opciones) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Este navegador solo permite la cámara en páginas HTTPS. " +
      "Abre el sistema con https:// y la IP de la laptop (ej. https://192.168.0.16:8443), " +
      "toca 'Avanzado' y luego 'Continuar al sitio'."
    );
  }
  return navigator.mediaDevices.getUserMedia(opciones);
}

function mensajeAmigableCamara(e) {
  if (e && e.message && e.message.includes("HTTPS")) return e.message;
  const nombre = e && e.name ? e.name : "";
  if (nombre === "NotReadableError" || nombre === "TrackStartError") {
    return "Esa cámara está ocupada por otra aplicación. Elige otra en el selector.";
  }
  if (nombre === "NotAllowedError") {
    return "No diste permiso de cámara al navegador. Actívalo en los permisos del sitio.";
  }
  if (nombre === "OverconstrainedError") {
    return "La cámara elegida no está disponible. Prueba con 'Automática' u otra del selector.";
  }
  return "No se pudo acceder a la cámara: " + (e && e.message ? e.message : nombre);
}

function opcionesDeVideoLocal(extra = {}) {
  return { facingMode: "user", ...extra };
}

function camarasActivas() {
  return estado.camaras.filter((c) => c.activa !== false);
}

function actualizarSelectorRegistro() {
  const sel = $("#selector_cam_reg");
  if (!sel) return;
  const previo = sel.value;
  sel.innerHTML = '<option value="">Cámara de este dispositivo</option>';
  camarasActivas().forEach((c) => {
    const o = document.createElement("option");
    o.value = "sys:" + c.id;
    o.textContent = `${c.nombre} (${c.id})`;
    sel.appendChild(o);
  });
  if ([...sel.options].some((o) => o.value === previo)) sel.value = previo;
}

/* ---------- Dashboard ---------- */

function conectarDashboard() {
  const protocolo = location.protocol === "https:" ? "wss" : "ws";
  wsDashboard = new WebSocket(`${protocolo}://${location.host}/ws/dashboard`);
  wsDashboard.onopen = () => refrescarEstado();
  wsDashboard.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.tipo === "camaras") {
      estado.camaras = msg.camaras;
      renderizarVideos();
    } else if (msg.tipo === "alerta") recibirAlerta(msg);
    else if (msg.tipo === "registro") notificarRegistro(msg);
    else if (msg.tipo === "alertas_historico") pintarAlertas(msg.alertas);
  };
  wsDashboard.onclose = () => setTimeout(conectarDashboard, 2500);
}

async function refrescarEstado() {
  try {
    const r = await fetch("/api/estado");
    const e = await r.json();
    $("#estado_sistema").innerHTML =
      `<i class="fa-solid fa-microchip"></i> Motor: ${e.motor_listo ? "listo" : "cargando…"} · ` +
      `<i class="fa-solid fa-database"></i> Supabase: ${e.supabase_configurado ? "conectado" : "SIN configurar"} · ` +
      `${e.camaras_conectadas} cámara(s) · ${e.personas_registradas} persona(s)`;
    $("#contador_personas").textContent = e.personas_registradas;
  } catch {}
}

setInterval(refrescarEstado, 5000);
setInterval(cargarStats, 2000);
setInterval(cargarColores, 10000);
setInterval(cargarUltimosRecs, 6000);

function ticRelojHud() {
  const el = $("#hud_reloj");
  if (el) el.textContent = new Date().toLocaleTimeString("es-ES", { hour12: false });
}
setInterval(ticRelojHud, 1000);
ticRelojHud();

async function cargarStats() {
  try {
    const r = await fetch("/api/estadisticas");
    pintarStats((await r.json()).personas);
  } catch {}
}

function principalElegida() {
  if (!camarasActivas().length) return null;
  return camarasActivas()[0];
}

function renderizarVideos() {
  const activas = camarasActivas();
  const firma = estado.camaras
    .map((c) => `${c.id}:${c.activa === false ? "off" : "on"}:${activas.includes(c) && !estaSinSenal(c) ? "1" : "0"}`)
    .join("|");
  const principal = principalElegida();
  const badgeVivo = $("#badge_envivo");
  if (badgeVivo) badgeVivo.classList.toggle("oculto", !!(principal && estaSinSenal(principal)));

  if (!principal) {
    $("#etiqueta_principal").innerHTML =
      '<i class="fa-solid fa-satellite-dish"></i> Esperando cámaras…';
    $("#video_principal").removeAttribute("src");
    estado.principalId = null;
  } else if (estaSinSenal(principal)) {
    if (estado.principalId === principal.id) {
      $("#video_principal").removeAttribute("src");
      estado.principalId = null;
    }
    $("#etiqueta_principal").innerHTML =
      `<i class="fa-solid fa-triangle-exclamation"></i> ${principal.nombre}: SIN SEÑAL`;
  } else if (principal && estado.principalId !== principal.id) {
    estado.principalId = principal.id;
    $("#video_principal").src = `/api/stream/${principal.id}`;
    $("#etiqueta_principal").innerHTML =
      `<i class="fa-solid fa-star"></i> ${principal.nombre} (${principal.id})`;
  }

  if (firma !== estado.firmaCamaras) {
    estado.firmaCamaras = firma;
    actualizarSelectorRegistro();
    pintarFilmstrip(principal);
    pintarGrid(principal);
  }
}

function pintarFilmstrip(principal) {
  const cont = $("#filmstrip");
  cont.innerHTML = "";
  camarasActivas()
    .filter((c) => (!principal || c.id !== principal.id) && !estaSinSenal(c))
    .forEach((cam) => {
    const div = document.createElement("div");
    div.className = "mini_cam";
    div.title = "Clic para hacer principal";
    div.onclick = () => hacerPrincipal(cam.id);
    const img = document.createElement("img");
    img.src = `/api/stream/${cam.id}`;
    img.loading = "lazy";
    const lbl = document.createElement("div");
    lbl.className = "nombre_mc";
    lbl.innerHTML = `<i class="fa-solid ${cam.tipo === "usb" ? "fa-plug" : "fa-globe"}"></i> ${cam.nombre}`;
    div.appendChild(img);
    div.appendChild(lbl);
    cont.appendChild(div);
  });
}

function pintarGrid(principal) {
  const grid = $("#grid_camaras");
  grid.innerHTML = "";
  if (!estado.camaras.length) {
    grid.innerHTML =
      '<p class="vacio"><i class="fa-solid fa-plug-circle-xmark"></i> No hay cámaras conectadas. Pulsa "Detectar cámaras" o conecta una webcam USB.</p>';
    return;
  }
  estado.camaras.forEach((cam) => {
    const esPrincipal = !!principal && cam.id === principal.id;
    const encendida = cam.activa !== false;
    const sinSenal = !encendida || estaSinSenal(cam);
    const t = document.createElement("div");
    t.className = "tarjeta_cam"
      + (esPrincipal ? " principal" : "")
      + (sinSenal ? " sin_senal" : "")
      + (!encendida ? " apagada" : "");
    const acciones = document.createElement("div");
    acciones.className = "acciones_cam";

    const btnEstado = document.createElement("button");
    if (encendida) {
      btnEstado.className = "secundario";
      btnEstado.innerHTML = '<i class="fa-solid fa-power-off"></i> Apagar';
      btnEstado.onclick = () => cambiarEstadoCam(cam.id, false);
    } else {
      btnEstado.className = "primario";
      btnEstado.innerHTML = '<i class="fa-solid fa-power-off"></i> Activar';
      btnEstado.onclick = () => cambiarEstadoCam(cam.id, true);
    }
    acciones.appendChild(btnEstado);

    if (encendida && !esPrincipal && !estaSinSenal(cam)) {
      const btnPrin = document.createElement("button");
      btnPrin.className = "secundario";
      btnPrin.innerHTML = '<i class="fa-solid fa-star"></i> Hacer principal';
      btnPrin.onclick = () => hacerPrincipal(cam.id);
      acciones.appendChild(btnPrin);
    }

    t.innerHTML = `
      <div class="preview_cam">${
        !encendida
          ? '<div class="overlay_sinsenal"><i class="fa-solid fa-plug-circle-xmark"></i> APAGADA</div>'
          : sinSenal
            ? '<div class="overlay_sinsenal"><i class="fa-solid fa-video-slash"></i> SIN SEÑAL</div>'
            : `<img src="/api/stream/${cam.id}" loading="lazy" alt="${cam.nombre}">`
      }</div>
      <span class="tipo_chip${!encendida ? " chip_apagada" : ""}">
        <i class="fa-solid ${cam.tipo === "usb" ? "fa-plug" : "fa-globe"}"></i>
        ${cam.tipo === "usb" ? "USB local" : "Navegador remoto"}${!encendida ? " · apagada" : ""}</span>
      <h3><span class="estado_dot${sinSenal ? " apagado" : ""}"></span>${cam.nombre}</h3>
      <code class="id_cam">${cam.id}</code>`;
    t.appendChild(acciones);

    if (esPrincipal && encendida) {
      const nota = document.createElement("span");
      nota.className = "nota_principal";
      nota.innerHTML = '<i class="fa-solid fa-star"></i> Cámara principal';
      t.appendChild(nota);
    }
    grid.appendChild(t);
  });
}

function haceCuanto(iso) {
  const seg = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (seg < 60) return `hace ${seg}s`;
  const min = Math.floor(seg / 60);
  if (min < 60) return `hace ${min} min`;
  const h = Math.floor(min / 60);
  return `hace ${h} h`;
}

function confPct(c) {
  if (typeof c !== "number" || isNaN(c)) return null;
  return Math.max(0, Math.min(100, c <= 1 ? c * 100 : c));
}

async function hacerPrincipal(cam_id) {
  await fetch("/api/camara_principal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cam_id }),
  });
  try {
    const r = await fetch("/api/camaras");
    estado.camaras = (await r.json()).camaras;
    estado.principalId = null;
    renderizarVideos();
  } catch {}
}

function pintarStats(personas) {
  const cont = $("#lista_stats");
  if (!personas || !personas.length) {
    cont.innerHTML = '<p class="vacio">Sin personas en escena</p>';
    return;
  }
  cont.innerHTML = "";
  personas.slice(0, 12).forEach((p) => {
    const color = coloresPersona[p.nombre] || "#22d3ee";
    const inicial = (p.nombre || "?").trim().charAt(0).toUpperCase() || "?";
    const pct = confPct(p.confianza);
    const d = document.createElement("div");
    d.className = "persona_stat" + (p.conocido ? " conocida" : "");
    d.innerHTML = `
      <div class="cabecera_stat">
        <div class="avatar_p" style="background:${color}">${inicial}</div>
        <div class="nombre_p">${p.nombre}</div>
      </div>
      <div class="fila_mini"><span><i class="fa-solid fa-ruler-vertical"></i> Distancia aprox:</span><b>${p.distancia_aprox_m ?? "?"} m</b></div>
      <div class="fila_mini"><span><i class="fa-solid fa-camera"></i> Cámara:</span><span>${p.camara}</span></div>
      <div class="fila_mini"><span><i class="fa-solid fa-clock"></i> Tiempo en escena:</span><span>${p.tiempo_en_escena_s}s</span></div>
      <div class="fila_mini"><span><i class="fa-solid fa-eye"></i> Veces detectado:</span><span>${p.veces_detectado}</span></div>
      ${pct != null
        ? `<div class="fila_mini"><span><i class="fa-solid fa-percent"></i> Confianza</span><span>${Math.round(pct)}%</span></div>
           <div class="barra_conf"><span style="width:${pct}%"></span></div>`
        : `<div class="fila_mini"><span><i class="fa-solid fa-percent"></i> Confianza:</span><span>${p.confianza ?? "-"}</span></div>`}`;
    cont.appendChild(d);
  });
}

/* ---------- Alertas discretas (mini-toast) ---------- */

function recibirAlerta(a) {
  estado.alertas += 1;
  const badge = $("#badge_alertas");
  badge.textContent = estado.alertas;
  badge.classList.remove("oculto");
  crearToast(a);
  sonarPitidoCorto();
  agregarAlertaLista(a);
}

function crearToast(a) {
  const cont = $("#toasts");
  while (cont.children.length >= 4) cont.firstChild.remove();

  const t = document.createElement("div");
  t.className = "toast_alerta";

  const icono = document.createElement("div");
  icono.className = "toast_icono";
  icono.innerHTML = '<i class="fa-solid fa-mobile-screen-button"></i>';

  const cuerpo = document.createElement("div");
  cuerpo.className = "toast_cuerpo";
  cuerpo.innerHTML = `
    <div class="toast_titulo"><i class="fa-solid fa-triangle-exclamation"></i> Posible replay</div>
    <div class="toast_detalle">${a.camara}<br>${a.detalle || ""}<br>${new Date().toLocaleTimeString()}</div>`;
  const btnVer = document.createElement("button");
  btnVer.className = "btn_ver_captura";
  btnVer.innerHTML = '<i class="fa-solid fa-image"></i> Ver captura';
  btnVer.onclick = () => abrirModalAlerta(a);
  cuerpo.appendChild(btnVer);

  const btnCerrar = document.createElement("button");
  btnCerrar.className = "cerrar_toast";
  btnCerrar.innerHTML = "&times;";
  btnCerrar.onclick = () => t.remove();

  t.appendChild(icono);
  t.appendChild(cuerpo);
  t.appendChild(btnCerrar);
  cont.appendChild(t);

  setTimeout(() => {
    t.classList.add("saliendo");
    setTimeout(() => t.remove(), 450);
  }, 8000);
}

function abrirModalAlerta(a) {
  $("#modal_alerta_detalle").innerHTML =
    `<i class="fa-solid fa-mobile-screen-button"></i> <b>${a.camara}</b><br>${a.detalle || ""}`;
  if (a.captura_b64) $("#modal_alerta_img").src = "data:image/jpeg;base64," + a.captura_b64;
  $("#modal_alerta").classList.remove("oculto");
}

function cerrarModalAlerta() {
  $("#modal_alerta").classList.add("oculto");
}
window.cerrarModalAlerta = cerrarModalAlerta;

function agregarAlertaLista(a) {
  const lista = $("#lista_alertas");
  if (lista.querySelector(".vacio")) lista.innerHTML = "";
  const t = document.createElement("div");
  t.className = "tarjeta_alerta";
  const imgHtml = a.captura_b64
    ? `<img src="data:image/jpeg;base64,${a.captura_b64}">`
    : '<img src="">';
  t.innerHTML = `
    ${imgHtml}
    <div>
      <div class="titulo_a"><i class="fa-solid fa-mobile-screen-button"></i> Posible replay con teléfono — ${a.camara}</div>
      <div class="info_a">${a.detalle || ""}<br>${new Date(a.creado_en || Date.now()).toLocaleString()}</div>
    </div>`;
  lista.prepend(t);
}

function pintarAlertas(alertas) {
  (alertas || []).slice().reverse().forEach(agregarAlertaLista);
}

function notificarRegistro() {
  refrescarEstado();
  cargarColores();
  cargarUltimosRecs();
}

function sonarPitidoCorto() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "sine";
    o.frequency.value = 720;
    g.gain.setValueAtTime(0.18, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.16);
    o.connect(g);
    g.connect(ctx.destination);
    o.start(ctx.currentTime);
    o.stop(ctx.currentTime + 0.18);
  } catch {}
}

/* ---------- Colores por persona ---------- */

async function cargarColores() {
  let data;
  try {
    data = await (await fetch("/api/personas")).json();
  } catch { return; }
  const cont = $("#lista_colores");
  if (!data.personas || !data.personas.length) {
    cont.innerHTML = '<p class="vacio">Aún no hay personas registradas</p>';
    return;
  }
  cont.innerHTML = "";
  coloresPersona = {};
  data.personas.forEach((p) => {
    coloresPersona[p.nombre] = p.color_hex || "#22d3ee";
    const fila = document.createElement("div");
    fila.className = "persona_color";

    const swatch = document.createElement("input");
    swatch.type = "color";
    swatch.value = p.color_hex || "#22d3ee";

    const etiqueta = document.createElement("label");
    etiqueta.textContent = p.nombre;

    const btnBorrar = document.createElement("button");
    btnBorrar.className = "btn_eliminar_persona";
    btnBorrar.title = `Eliminar a ${p.nombre}`;
    btnBorrar.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
    btnBorrar.onclick = () => eliminarPersona(p.nombre);

    fila.appendChild(swatch);
    fila.appendChild(etiqueta);
    fila.appendChild(btnBorrar);
    cont.appendChild(fila);

    swatch.onchange = async () => {
      try {
        await fetch(`/api/color_persona/${encodeURIComponent(p.nombre)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ color: swatch.value }),
        });
        etiqueta.classList.add("guardado");
        setTimeout(() => etiqueta.classList.remove("guardado"), 1200);
      } catch {}
    };
  });
}

async function eliminarPersona(nombre) {
  if (!confirm(`¿Eliminar a "${nombre}"? Se borrarán su rostro, sus fotos y sus datos de la base.`)) return;
  try {
    await fetch(`/api/persona/${encodeURIComponent(nombre)}`, { method: "DELETE" });
    refrescarEstado();
    cargarColores();
    cargarUltimosRecs();
  } catch {}
}

/* ---------- Últimos reconocimientos (Supabase) ---------- */

async function cargarUltimosRecs() {
  let data;
  try {
    data = await (await fetch("/api/ultimos_reconocimientos")).json();
  } catch { return; }
  const cont = $("#ultimos_recs");
  const recs = data.reconocimientos || [];
  if (!recs.length) {
    cont.innerHTML = '<p class="vacio">Sin historial aún</p>';
    return;
  }
  cont.innerHTML = "";
  recs.forEach((r) => {
    const d = document.createElement("div");
    d.className = "rec_historico";
    d.innerHTML = `
      <div class="nombre_p"><i class="fa-solid fa-circle-user"></i> ${r.nombre_detectado || "?"}</div>
      <div class="fila_mini">
        <span><i class="fa-solid fa-camera"></i> ${r.camara || "-"} · ${haceCuanto(r.creado_en)}</span>
        <b>${r.distancia_aprox_m != null ? `~${Number(r.distancia_aprox_m).toFixed(1)} m` : ""}</b>
      </div>`;
    cont.appendChild(d);
  });
}

/* ---------- Pestañas ---------- */

document.querySelectorAll(".pestana").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".pestana").forEach((b) => b.classList.remove("activa"));
    btn.classList.add("activa");
    document.querySelectorAll(".tab").forEach((s) => s.classList.remove("activa"));
    $("#tab_" + btn.dataset.tab).classList.add("activa");
    if (btn.dataset.tab === "alertas") {
      estado.alertas = 0;
      const b = $("#badge_alertas");
      b.textContent = "0";
      b.classList.add("oculto");
    }
    if (btn.dataset.tab === "registro") { cargarColores(); actualizarSelectorRegistro(); }
    if (btn.dataset.tab === "reconocimiento") cargarUltimosRecs();
  };
});

/* ---------- Registro facial (multi-persona, dual fuente) ---------- */

async function activarCamaraRegistro() {
  const res = $("#resultado_registro");
  const selVal = $("#selector_cam_reg").value;

  if (selVal.startsWith("sys:")) {
    modoCapturaRegistro = selVal.slice(4);
    if (streamRegistro) {
      streamRegistro.getTracks().forEach((t) => t.stop());
      streamRegistro = null;
    }
    $("#video_registro").classList.add("oculto");
    $("#img_registro_sys").src = `/api/stream/${modoCapturaRegistro}`;
    $("#img_registro_sys").classList.remove("oculto");
    $("#btn_registrar").disabled = false;
    res.textContent = "Usando la cámara del sistema (" + modoCapturaRegistro + ").";
    res.className = "resultado ok";
    return;
  }

  modoCapturaRegistro = "";
  $("#img_registro_sys").classList.add("oculto");
  $("#video_registro").classList.remove("oculto");
  try {
    streamRegistro = await pedirCamara({
      video: opcionesDeVideoLocal({ width: { ideal: 720 } }),
      audio: false,
    });
    $("#video_registro").srcObject = streamRegistro;
    $("#btn_registrar").disabled = false;
    res.textContent = "Cámara activa. Colócate frente a ella.";
    res.className = "resultado ok";
  } catch (e) {
    res.textContent = mensajeAmigableCamara(e);
    res.className = "resultado error";
  }
}

function fuenteCapturaRegistro() {
  return modoCapturaRegistro ? $("#img_registro_sys") : $("#video_registro");
}

async function enviarRegistro() {
  const nombre = $("#reg_nombre").value.trim();
  const res = $("#resultado_registro");
  if (!nombre) {
    res.textContent = "Escribe tu nombre para registrarte.";
    res.className = "resultado error";
    return;
  }
  const fuente = fuenteCapturaRegistro();
  const ancho = fuente.videoWidth || fuente.naturalWidth;
  const alto = fuente.videoHeight || fuente.naturalHeight;
  if (!ancho) {
    res.textContent = "La imagen de la cámara aún carga… intenta de nuevo en un segundo.";
    res.className = "resultado error";
    return;
  }
  if (!modoCapturaRegistro && !streamRegistro) {
    res.textContent = "Primero activa tu cámara.";
    res.className = "resultado error";
    return;
  }
  res.className = "resultado ok";
  res.textContent = "Capturando rostros… mantén la mirada fija.";
  const canvas = $("#canvas_registro");
  canvas.width = ancho;
  canvas.height = alto;
  const ctx2d = canvas.getContext("2d");
  const imagenes = [];
  for (let i = 0; i < 10; i++) {
    ctx2d.drawImage(fuente, 0, 0, canvas.width, canvas.height);
    imagenes.push(canvas.toDataURL("image/jpeg", 0.85));
    await new Promise((r) => setTimeout(r, 350));
  }
  try {
    const r = await fetch("/api/registrar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre, codigo: "", imagenes }),
    });
    const data = await r.json();
    if (data.ok) {
      res.className = "resultado ok";
      res.innerHTML = `<i class="fa-solid fa-circle-check"></i> ¡Listo, ${nombre}! ${data.muestras} muestras guardadas. Puedes registrar a otra persona.`;
      $("#reg_nombre").value = "";
      $("#reg_nombre").focus();
      refrescarEstado();
      cargarColores();
    } else {
      res.className = "resultado error";
      res.textContent = data.error || "Error al registrar.";
    }
  } catch (e) {
    res.className = "resultado error";
    res.textContent = "Error de conexión: " + e.message;
  }
}

/* ---------- Modo pantalla completa ---------- */

function alternarFullscreen() {
  const zona = document.querySelector(".zona_principal");
  if (!document.fullscreenElement) {
    if (zona.requestFullscreen) zona.requestFullscreen();
  } else {
    if (document.exitFullscreen) document.exitFullscreen();
  }
}

document.addEventListener("fullscreenchange", () => {
  const en = !!document.fullscreenElement;
  const b = $("#btn_fullscreen");
  if (b) b.innerHTML = `<i class="fa-solid ${en ? "fa-compress" : "fa-expand"}"></i>`;
});

/* ---------- Diagnóstico de reconocimiento ---------- */

async function probarReconocimiento() {
  const salida = $("#resultado_prueba");
  const principal = principalElegida() || camarasActivas()[0];
  if (!principal) {
    salida.innerHTML = '<span class="error_txt">No hay cámaras conectadas.</span>';
    return;
  }
  salida.textContent = "Analizando " + principal.nombre + "…";
  try {
    const r = await fetch(`/api/probar_reconocimiento/${principal.id}`);
    const d = await r.json();
    if (!d.ok) {
      salida.innerHTML = `<span class="error_txt">${d.error || "Sin imagen de esa cámara."}</span>`;
      return;
    }
    if (!d.rostros.length) {
      salida.innerHTML = '<span class="error_txt">Ningún rostro visible ahora. Ponte frente a la cámara con buena luz.</span>';
      return;
    }
    salida.innerHTML = d.rostros
      .map((f) =>
        f.conocido
          ? `<span class="ok_txt">✔ ${f.nombre}: sim ${f.similitud.toFixed(3)} (umbral ${d.umbral})</span>`
          : `<span class="warn_txt">? Desconocido: mejor coincidencia "${f.mejor_nombre}" sim ${f.mejor_sim.toFixed(3)} vs umbral ${d.umbral}</span>`
      )
      .join("<br>");
  } catch {
    salida.innerHTML = '<span class="error_txt">Error consultando el servidor.</span>';
  }
}

/* ---------- Controles de cámaras ---------- */

async function controlarCamaras(url, texto) {
  const aviso = $("#estado_controles_cam");
  aviso.textContent = texto;
  try {
    const r = await fetch(url, { method: "POST" });
    const data = await r.json();
    if (data.camaras) {
      estado.camaras = data.camaras;
      renderizarVideos();
    }
    aviso.textContent = "Listo.";
  } catch {
    aviso.textContent = "Error al ejecutar.";
  }
  setTimeout(() => { aviso.textContent = ""; }, 2500);
}

async function cambiarEstadoCam(cam_id, activa) {
  const aviso = $("#estado_controles_cam");
  aviso.textContent = activa ? "Activando cámara…" : "Apagando cámara…";
  try {
    const r = await fetch(`/api/camara/${cam_id}/estado`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ activa }),
    });
    const data = await r.json();
    if (data.camaras) {
      estado.camaras = data.camaras;
      estado.principalId = null;
      renderizarVideos();
    }
    aviso.textContent = "Listo.";
  } catch {
    aviso.textContent = "Error al cambiar el estado.";
  }
  setTimeout(() => { aviso.textContent = ""; }, 2500);
}

$("#btn_activar_cam_reg").onclick = activarCamaraRegistro;
$("#btn_registrar").onclick = enviarRegistro;
$("#selector_cam_reg").onchange = () => activarCamaraRegistro();
$("#btn_fullscreen").onclick = alternarFullscreen;
$("#btn_probar_rec").onclick = probarReconocimiento;
$("#btn_reescanear").onclick = () => controlarCamaras("/api/camaras/reescanear", "Buscando cámaras…");
$("#btn_reiniciar_usb").onclick = () =>
  controlarCamaras("/api/camaras/reiniciar_usb", "Reiniciando cámaras USB…");

conectarDashboard();
cargarColores();
cargarUltimosRecs();
