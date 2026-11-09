const $ = (sel) => document.querySelector(sel);

let streamMovil = null;
let modoCapturaMovil = "";
let nombreActual = "";
let temporizadorActividad = null;

function pedirCamara(opciones) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Este navegador solo permite la cámara en páginas HTTPS. " +
      "Abre con https:// y la IP de la laptop (ej. https://192.168.0.16:8443), " +
      "toca 'Avanzado' y luego 'Continuar al sitio'."
    );
  }
  return navigator.mediaDevices.getUserMedia(opciones);
}

async function llenarSelectorCameras() {
  const selectEl = $("#selector_cam_movil");
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    selectEl.innerHTML = '<option value="">(requiere HTTPS)</option>';
    return;
  }
  let camarasSistema = [];
  try {
    const r = await fetch("/api/camaras");
    camarasSistema = (await r.json()).camaras || [];
  } catch {}
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter((d) => d.kind === "videoinput");
    const previo = selectEl.value;
    const g1 = document.createElement("optgroup");
    g1.label = "Este teléfono";
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "Automática (frontal)";
    g1.appendChild(auto);
    cams.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Cámara ${i + 1}`;
      g1.appendChild(o);
    });
    const g2 = document.createElement("optgroup");
    g2.label = "Cámaras del sistema";
    camarasSistema.forEach((c) => {
      const o = document.createElement("option");
      o.value = "sys:" + c.id;
      o.textContent = `${c.nombre} (${c.id})`;
      g2.appendChild(o);
    });
    selectEl.innerHTML = "";
    selectEl.appendChild(g1);
    if (camarasSistema.length) selectEl.appendChild(g2);
    if (previo && [...selectEl.querySelectorAll("option")].some((o) => o.value === previo)) {
      selectEl.value = previo;
    }
  } catch {}
}

function haceCuanto(iso) {
  const seg = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (seg < 60) return `hace ${seg} segundos`;
  const min = Math.floor(seg / 60);
  if (min < 60) return `hace ${min} minutos`;
  const h = Math.floor(min / 60);
  if (h < 24) return `hace ${h} horas`;
  const d = Math.floor(h / 24);
  return `hace ${d} día(s)`;
}

async function activarCamara() {
  const res = $("#resultado_movil");
  const selVal = $("#selector_cam_movil").value;
  try {
    if (selVal.startsWith("sys:")) {
      modoCapturaMovil = selVal.slice(4);
      if (streamMovil) {
        streamMovil.getTracks().forEach((t) => t.stop());
        streamMovil = null;
      }
      $("#video_movil").classList.add("oculto");
      $("#img_movil_sys").src = `/api/stream/${modoCapturaMovil}`;
      $("#img_movil_sys").classList.remove("oculto");
      $("#btn_reg_movil").disabled = false;
      res.textContent = "Usando la cámara del sistema (" + modoCapturaMovil + ").";
      res.style.color = "var(--verde)";
      return;
    }
    modoCapturaMovil = "";
    $("#img_movil_sys").classList.add("oculto");
    $("#video_movil").classList.remove("oculto");
    const id = selVal;
    const videoOps = id
      ? { deviceId: { exact: id }, width: { ideal: 720 } }
      : { facingMode: "user", width: { ideal: 720 } };
    streamMovil = await pedirCamara({ video: videoOps, audio: false });
    $("#video_movil").srcObject = streamMovil;
    $("#placeholder_cam").classList.add("oculto");
    $("#btn_reg_movil").disabled = false;
    res.textContent = "Cámara lista.";
    res.className = "resultado_movil";
    res.style.color = "var(--verde)";
    llenarSelectorCameras();
  } catch (e) {
    res.textContent = mensajeAmigableCamara(e);
    res.className = "resultado_movil error";
    res.style.color = "var(--rojo)";
  }
}

function mensajeAmigableCamara(e) {
  if (e && e.message && e.message.includes("HTTPS")) return e.message;
  const nombre = e && e.name ? e.name : "";
  if (nombre === "NotReadableError" || nombre === "TrackStartError") {
    return "Esa cámara está ocupada por otra app. Elige otra en el selector.";
  }
  if (nombre === "NotAllowedError") {
    return "No diste permiso de cámara al navegador. Actívalo en los permisos del sitio.";
  }
  if (nombre === "OverconstrainedError") {
    return "La cámara elegida no está disponible. Prueba con 'Automática' u otra del selector.";
  }
  return "No se pudo acceder a la cámara: " + (e && e.message ? e.message : nombre);
}

async function registrarme() {
  const nombre = $("#nombre_movil").value.trim();
  const res = $("#resultado_movil");
  if (!nombre) {
    res.textContent = "Escribe tu nombre primero.";
    res.style.color = "var(--rojo)";
    return;
  }
  const fuente = modoCapturaMovil ? $("#img_movil_sys") : $("#video_movil");
  const ancho = fuente.videoWidth || fuente.naturalWidth;
  const alto = fuente.videoHeight || fuente.naturalHeight;
  if (!ancho) {
    res.textContent = "La imagen de la cámara aún carga… intenta de nuevo en un segundo.";
    res.style.color = "var(--rojo)";
    return;
  }
  if (!modoCapturaMovil && !streamMovil) {
    res.textContent = "Primero activa tu cámara.";
    res.style.color = "var(--rojo)";
    return;
  }
  res.textContent = "Capturando tu rostro… mira fijo a la cámara.";
  res.style.color = "var(--texto)";

  const canvas = $("#canvas_movil");
  canvas.width = ancho;
  canvas.height = alto;
  const ctx = canvas.getContext("2d");
  const imagenes = [];
  for (let i = 0; i < 10; i++) {
    ctx.drawImage(fuente, 0, 0, canvas.width, canvas.height);
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
      res.innerHTML =
        `<i class="fa-solid fa-circle-check"></i> ¡Listo, ${nombre}! Se guardaron ${data.muestras} muestras. ` +
        "Puedes registrar a otra persona.";
      res.style.color = "var(--verde)";
      $("#nombre_movil").value = "";
      $("#nombre_movil").focus();
    } else {
      res.textContent = data.error || "No se pudo registrar.";
      res.style.color = "var(--rojo)";
    }
  } catch (e) {
    res.textContent = "Error de conexión: " + e.message;
    res.style.color = "var(--rojo)";
  }
}

async function verMiActividad() {
  const nombre = $("#nombre_movil").value.trim();
  const res = $("#resultado_movil");
  if (!nombre) {
    res.textContent = "Escribe tu nombre para buscar tu actividad.";
    res.style.color = "var(--rojo)";
    return;
  }
  try {
    const r = await fetch(`/api/mis_detecciones/${encodeURIComponent(nombre)}`);
    const data = await r.json();
    if (!data.detecciones || !data.detecciones.length) {
      res.textContent = `"${nombre}" todavía no tiene detecciones. Regístrate o pasa frente a una cámara.`;
      res.style.color = "var(--texto2)";
      return;
    }
    res.textContent = "";
    mostrarActividad(nombre);
  } catch {
    res.textContent = "Error de conexión.";
    res.style.color = "var(--rojo)";
  }
}

async function refrescarDetecciones() {
  if (!nombreActual) return;
  let data;
  try {
    data = await (await fetch(`/api/mis_detecciones/${encodeURIComponent(nombreActual)}`)).json();
  } catch { return; }
  const cont = $("#lista_detecciones");
  const dets = data.detecciones || [];
  if (!dets.length) {
    cont.innerHTML = '<div class="sin_datos">Aún no te han visto las cámaras. Pasa frente a una.</div>';
    return;
  }
  cont.innerHTML = "";
  dets.forEach((d) => {
    const item = document.createElement("div");
    item.className = "deteccion_item";
    const cuando = document.createElement("div");
    cuando.className = "cuando";
    cuando.innerHTML = `<i class="fa-solid fa-eye"></i> Visto ${haceCuanto(d.creado_en)}`;
    const donde = document.createElement("div");
    donde.className = "donde";
    donde.innerHTML =
      `<i class="fa-solid fa-camera"></i> Cámara: ${d.camara}` +
      (d.distancia_aprox_m != null ? ` · a ~${Number(d.distancia_aprox_m).toFixed(1)} m` : "");
    item.appendChild(cuando);
    item.appendChild(donde);
    cont.appendChild(item);
  });
}

function mostrarActividad(nombre, mensajeSaludo) {
  nombreActual = nombre;
  $("#paso_registro").classList.add("oculto");
  $("#paso_actividad").classList.remove("oculto");
  $("#titulo_actividad").innerHTML =
    `<i class="fa-solid fa-circle-user"></i> Hola, ${nombre}` +
    (mensajeSaludo ? `<br><small style="font-size:13px;color:var(--texto2)">${mensajeSaludo}</small>` : "");
  clearInterval(temporizadorActividad);
  refrescarDetecciones();
  temporizadorActividad = setInterval(refrescarDetecciones, 5000);
}

$("#btn_cam_movil").onclick = activarCamara;
$("#selector_cam_movil").onchange = () => {
  if (streamMovil || modoCapturaMovil) activarCamara();
};
$("#btn_reg_movil").onclick = registrarme;
$("#btn_ver_actividad").onclick = verMiActividad;
$("#btn_otra_persona").onclick = () => {
  clearInterval(temporizadorActividad);
  $("#paso_actividad").classList.add("oculto");
  $("#paso_registro").classList.remove("oculto");
};

llenarSelectorCameras();
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", llenarSelectorCameras);
}
