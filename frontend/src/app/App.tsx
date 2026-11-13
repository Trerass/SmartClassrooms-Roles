import { useEffect, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import { Bot, Building2, ImageUp, Loader2, Play, Send, Trash2, Wrench } from "lucide-react";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { cn } from "./components/ui/utils";

const SALONES = [
  { id: "38-101", nombre: "Aula 38-101", capacidad: 40 },
  { id: "38-102", nombre: "Aula 38-102", capacidad: 35 },
  { id: "38-201", nombre: "Aula 38-201", capacidad: 45 },
];

const TIPOS = ["image/jpeg", "image/png", "image/webp"];
const LIMITE_MS = 180_000; // sin GPU una foto grande tarda, pero nunca tanto

interface Deteccion {
  rol: string;
  color: string;
  confianza: number;
  caja: [number, number, number, number];
  objetos: string[];
}

interface Resultado {
  total: number;
  ancho: number;
  alto: number;
  roles: { rol: string; color: string; personas: number }[];
  detecciones: Deteccion[];
}

interface Foto {
  archivo: File;
  url: string;
  resultado?: Resultado;
  visibles?: boolean[];
}

type Salon = (typeof SALONES)[number];

async function analizar(archivo: File, salon: string): Promise<Resultado> {
  const form = new FormData();
  form.append("foto", archivo);
  form.append("salon", salon); // el agente responde con el último conteo de cada salón
  let respuesta: Response;
  try {
    respuesta = await fetch("/analizar", { method: "POST", body: form, signal: AbortSignal.timeout(LIMITE_MS) });
  } catch {
    throw new Error("No se pudo conectar con el servicio de análisis.");
  }
  if (!respuesta.ok) {
    const cuerpo = await respuesta.json().catch(() => null);
    // Sin detail: la API sigue cargando los modelos o está apagada.
    throw new Error(cuerpo?.detail ?? "El servicio de análisis no está listo. Inténtalo en un minuto.");
  }
  return respuesta.json();
}

export default function App() {
  const [vista, setVista] = useState<"asistente" | "fotos">("asistente");
  const [salonId, setSalonId] = useState(SALONES[0].id);
  const [fotos, setFotos] = useState<Record<string, Foto>>({});
  const [analizando, setAnalizando] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  // Los botones aparecen y desaparecen; sin esto, quien usa teclado pierde su lugar.
  const estadoRef = useRef<HTMLParagraphElement>(null);

  const salon = SALONES.find((s) => s.id === salonId)!;
  const foto = fotos[salonId];
  const resultado = foto?.resultado;

  const alElegirFoto = (e: ChangeEvent<HTMLInputElement>) => {
    const archivo = e.target.files?.[0];
    e.target.value = ""; // permite volver a elegir la misma foto
    if (!archivo) return;
    if (!TIPOS.includes(archivo.type)) {
      setError("Formato no admitido. Usa JPG, PNG o WebP.");
      return;
    }
    setError("");
    setFotos((prev) => ({ ...prev, [salonId]: { archivo, url: URL.createObjectURL(archivo) } }));
    estadoRef.current?.focus();
  };

  const procesar = async () => {
    setAnalizando(true);
    setError("");
    try {
      const r = await analizar(foto.archivo, salonId);
      setFotos((prev) => ({ ...prev, [salonId]: { ...foto, resultado: r, visibles: r.detecciones.map(() => true) } }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAnalizando(false);
      estadoRef.current?.focus();
    }
  };

  const quitar = () => {
    // El agente responde con el último conteo de cada salón: sin foto, ya no debe usarlo.
    if (foto.resultado) fetch(`/analisis/${salonId}`, { method: "DELETE" }).catch(() => {});
    URL.revokeObjectURL(foto.url);
    setError("");
    setFotos(({ [salonId]: _quitada, ...resto }) => resto);
    estadoRef.current?.focus();
  };

  const ayuda = !foto
    ? "Elige una foto para este salón."
    : analizando
      ? "Puede tardar unos segundos."
      : resultado
        ? "Quita la foto para probar con otra."
        : "Pulsa Procesar para analizar la foto.";

  const procesados = SALONES.filter((s) => fotos[s.id]?.resultado).length;

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 via-white to-white">
      <header className="sticky top-0 z-10 border-b border-gray-200/70 bg-white/80 backdrop-blur">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <Building2 className="w-7 h-7 sm:w-8 sm:h-8 text-blue-600 shrink-0" aria-hidden />
            <div className="min-w-0">
              <h1 className="truncate text-base sm:text-lg font-bold text-gray-900 leading-tight">Ocupación de salones</h1>
              <p className="hidden sm:block text-xs text-gray-500">Universidad EAFIT</p>
            </div>
          </div>
          <nav className="flex shrink-0 rounded-full bg-gray-100 p-1 text-sm" aria-label="Secciones">
            {(
              [
                ["asistente", "Asistente"],
                ["fotos", `Fotos (${procesados}/${SALONES.length})`],
              ] as const
            ).map(([id, texto]) => (
              <button
                key={id}
                type="button"
                aria-pressed={vista === id}
                onClick={() => setVista(id)}
                className={cn(
                  "rounded-full px-3 sm:px-4 py-1.5 font-medium transition-colors",
                  vista === id ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-900",
                )}
              >
                {texto}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 sm:px-6 py-10">
        {/* Las dos vistas siguen montadas: cambiar de pestaña no pierde la conversación ni las fotos. */}
        <div className={cn(vista !== "asistente" && "hidden")}>
          <Asistente salones={SALONES.map((s) => ({ id: s.id, procesado: !!fotos[s.id]?.resultado }))} onIrAFotos={() => setVista("fotos")} />
        </div>

        <section className={cn("space-y-5", vista !== "fotos" && "hidden")} aria-labelledby="titulo-fotos">
          <div>
            <h2 id="titulo-fotos" className="text-2xl font-semibold text-gray-900">
              Fotos de los salones
            </h2>
            <p className="text-sm text-gray-500">
              Elige un salón, sube su foto y pulsa Procesar. El asistente responde con el último conteo de cada salón.
            </p>
          </div>
          <Card className="p-6 gap-6">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {SALONES.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  aria-pressed={s.id === salonId}
                  disabled={analizando}
                  onClick={() => {
                    setSalonId(s.id);
                    setError("");
                  }}
                  className={cn(
                    "rounded-xl border bg-white p-4 text-left transition-colors disabled:opacity-60",
                    s.id === salonId ? "border-blue-600 ring-1 ring-blue-600" : "border-gray-200 hover:border-gray-400",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-semibold text-gray-900">{s.nombre}</p>
                    <span
                      className={cn(
                        "rounded-full px-2 py-0.5 text-xs font-medium",
                        fotos[s.id]?.resultado ? "bg-green-50 text-green-700" : "bg-gray-100 text-gray-500",
                      )}
                    >
                      {fotos[s.id]?.resultado ? "Procesada" : "Sin datos"}
                    </span>
                  </div>
                  {fotos[s.id] && !fotos[s.id].resultado && <p className="mt-1 text-sm text-gray-500">Foto sin procesar</p>}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <input
                ref={inputRef}
                type="file"
                accept={TIPOS.join(",")}
                className="hidden"
                onChange={alElegirFoto}
                aria-label={`Foto de ${salon.nombre}`}
              />
              {!foto && (
                <Button onClick={() => inputRef.current?.click()}>
                  <ImageUp aria-hidden />
                  Subir foto
                </Button>
              )}
              {foto && !resultado && (
                <Button onClick={procesar} disabled={analizando}>
                  {analizando ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
                  {analizando ? "Procesando…" : "Procesar"}
                </Button>
              )}
              {foto && (
                <Button variant="outline" onClick={quitar} disabled={analizando}>
                  <Trash2 aria-hidden />
                  Quitar foto
                </Button>
              )}
              <p
                ref={estadoRef}
                role="status"
                tabIndex={-1}
                className={cn("text-sm outline-none", error ? "text-red-600" : "text-gray-500")}
              >
                {error || ayuda}
              </p>
            </div>

            {foto && !resultado && (
              <img src={foto.url} alt={`Foto de ${salon.nombre} sin procesar`} className="w-full rounded-xl border border-gray-200" />
            )}
            {foto && resultado && (
              <VistaResultado
                salon={salon}
                url={foto.url}
                resultado={resultado}
                visibles={foto.visibles!}
                onVisibles={(visibles) => setFotos((prev) => ({ ...prev, [salonId]: { ...prev[salonId], visibles } }))}
              />
            )}
          </Card>
        </section>
      </main>
    </div>
  );
}

function VistaResultado({
  salon,
  url,
  resultado,
  visibles,
  onVisibles,
}: {
  salon: Salon;
  url: string;
  resultado: Resultado;
  visibles: boolean[];
  onVisibles: (visibles: boolean[]) => void;
}) {
  const [activa, setActiva] = useState<number | null>(null);
  const { ancho, alto, detecciones } = resultado;
  const letra = Math.max(ancho, alto) / 45;
  const area = (d: Deteccion) => (d.caja[2] - d.caja[0]) * (d.caja[3] - d.caja[1]);
  // Las cajas grandes se dibujan primero: así una pequeña dentro de otra se puede seleccionar.
  const orden = detecciones.map((_, i) => i).sort((a, b) => area(detecciones[b]) - area(detecciones[a]));

  const elegirCaja = (i: number) => {
    setActiva(i);
    document.getElementById(`deteccion-${i}`)?.scrollIntoView?.({ block: "nearest" });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_18rem] gap-6 items-start">
      <div className="relative">
        <img
          src={url}
          alt={`Foto de ${salon.nombre} con ${resultado.total} personas detectadas`}
          className="w-full rounded-xl border border-gray-200"
        />
        {/* La lista de la derecha es la forma accesible de elegir; los recuadros son un atajo con el mouse. */}
        <svg viewBox={`0 0 ${ancho} ${alto}`} className="absolute inset-0 w-full h-full" aria-hidden>
          {orden
            .filter((i) => visibles[i])
            .map((i) => {
              const d = detecciones[i];
              const [x1, y1, x2, y2] = d.caja;
              return (
                <g key={i} data-deteccion={i} className="cursor-pointer" onClick={() => elegirCaja(i)} onMouseEnter={() => setActiva(i)}>
                  <rect
                    x={x1}
                    y={y1}
                    width={x2 - x1}
                    height={y2 - y1}
                    fill={d.color}
                    fillOpacity={i === activa ? 0.25 : 0}
                    stroke={d.color}
                    strokeWidth={i === activa ? 4 : 2}
                    vectorEffect="non-scaling-stroke"
                  />
                  <text
                    x={x1 + letra * 0.3}
                    y={y1 + letra}
                    fontSize={letra}
                    fontWeight="bold"
                    fill="white"
                    stroke="black"
                    strokeWidth={letra / 8}
                    paintOrder="stroke"
                  >
                    {i + 1}
                  </text>
                </g>
              );
            })}
        </svg>
      </div>

      <div className="space-y-4">
        <Card>
          <CardContent className="p-6 space-y-4">
            <div>
              <p className="text-sm text-gray-500">Personas</p>
              <p className="text-4xl font-bold text-gray-900">{resultado.total}</p>
              <p className="text-sm text-gray-500">{Math.round((resultado.total / salon.capacidad) * 100)}% de la capacidad</p>
            </div>
            <ul className="space-y-2">
              {resultado.roles.map((r) => (
                <li key={r.rol} className="flex items-center gap-2 text-gray-900">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: r.color }} aria-hidden />
                  <span className="flex-1">{r.rol}</span>
                  <span className="font-semibold tabular-nums">{r.personas}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {detecciones.length > 0 && (
          <Card>
            <CardContent className="p-4 space-y-3">
              <div className="flex items-center gap-2">
                <p className="flex-1 font-semibold text-gray-900">Recuadros</p>
                <Button size="sm" variant="outline" onClick={() => onVisibles(detecciones.map(() => true))}>
                  Todos
                </Button>
                <Button size="sm" variant="outline" onClick={() => onVisibles(detecciones.map(() => false))}>
                  Ninguno
                </Button>
              </div>
              <ul className="max-h-80 overflow-y-auto space-y-1">
                {detecciones.map((d, i) => (
                  <li
                    key={i}
                    id={`deteccion-${i}`}
                    onMouseEnter={() => setActiva(i)}
                    className={cn("rounded-md px-2 py-1", i === activa && "bg-blue-50")}
                  >
                    <label className="flex items-center gap-2 text-sm text-gray-900 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={visibles[i]}
                        onChange={() => onVisibles(visibles.map((v, j) => (j === i ? !v : v)))}
                        onFocus={() => setActiva(i)}
                      />
                      <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: d.color }} aria-hidden />
                      <span className="flex-1">
                        {i + 1}. {d.rol}
                      </span>
                      <span className="font-semibold tabular-nums">{Math.round(d.confianza * 100)}%</span>
                    </label>
                    {d.objetos.length > 0 && <p className="pl-11 text-xs text-gray-500">Con: {d.objetos.join(", ")}</p>}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

interface Paso {
  herramienta: string;
  argumentos: unknown;
  resultado: unknown;
}

interface Conversacion {
  pregunta: string;
  respuesta?: string;
  pasos?: Paso[];
  error?: string;
}

const EJEMPLOS = ["¿Qué salones tienen foto analizada?", "¿Qué salón está más lleno?", "¿Hay personal de seguridad en algún salón?"];

function Puntos() {
  return (
    <span className="flex items-center gap-1 py-1.5" aria-hidden>
      {[0, 150, 300].map((d) => (
        <span key={d} className="h-2 w-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: `${d}ms` }} />
      ))}
    </span>
  );
}

function Asistente({ salones, onIrAFotos }: { salones: { id: string; procesado: boolean }[]; onIrAFotos: () => void }) {
  const [pregunta, setPregunta] = useState("");
  const [historial, setHistorial] = useState<Conversacion[]>([]);
  const [pensando, setPensando] = useState<string | null>(null);
  const finRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (historial.length || pensando) finRef.current?.scrollIntoView?.({ behavior: "smooth", block: "nearest" });
  }, [historial, pensando]);

  const enviar = async (texto: string) => {
    const q = texto.trim();
    if (!q || pensando) return;
    setPregunta("");
    setPensando(q);
    let c: Conversacion;
    try {
      const r = await fetch("/preguntar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta: q }),
        signal: AbortSignal.timeout(LIMITE_MS),
      });
      const cuerpo = await r.json().catch(() => null);
      c =
        r.ok && cuerpo?.respuesta !== undefined
          ? { pregunta: q, respuesta: cuerpo.respuesta, pasos: cuerpo.pasos ?? [] }
          : { pregunta: q, error: cuerpo?.detail ?? "El asistente no está disponible." };
    } catch {
      c = { pregunta: q, error: "No se pudo conectar con el asistente." };
    }
    setHistorial((h) => [...h, c]);
    setPensando(null);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="text-center space-y-3">
        <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-600 text-white shadow-lg shadow-blue-600/20">
          <Bot className="w-6 h-6" aria-hidden />
        </span>
        <h2 className="text-2xl sm:text-3xl font-semibold tracking-tight text-gray-900">¿Qué quieres saber de los salones?</h2>
        <p className="text-sm text-gray-500">
          Pregunta por las personas, los roles o la ocupación. El asistente consulta los conteos de las fotos procesadas.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3 pt-1 text-xs">
          <ul className="flex items-center gap-4 text-gray-600" aria-label="Salones con foto procesada">
            {salones.map((s) => (
              <li key={s.id} className="inline-flex items-center gap-1.5">
                <span
                  className={cn("h-2 w-2 rounded-full transition-colors", s.procesado ? "bg-green-500" : "bg-gray-300")}
                  aria-hidden
                />
                {s.id}
                <span className="sr-only">{s.procesado ? "con foto procesada" : "sin foto"}</span>
              </li>
            ))}
          </ul>
          <button
            type="button"
            onClick={onIrAFotos}
            className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2.5 py-1 text-xs font-medium text-blue-700 transition-colors hover:bg-blue-100"
          >
            <ImageUp className="w-3.5 h-3.5" aria-hidden />
            Subir fotos
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm">
        <ul className="space-y-5 p-5 sm:p-6" aria-live="polite">
          {historial.length === 0 && !pensando && (
            <li className="grid gap-2 sm:grid-cols-3">
              {EJEMPLOS.map((e) => (
                <button
                  key={e}
                  type="button"
                  onClick={() => enviar(e)}
                  className="rounded-xl border border-gray-200 p-3 text-left text-sm text-gray-700 transition-colors hover:border-blue-300 hover:bg-blue-50/60"
                >
                  {e}
                </button>
              ))}
            </li>
          )}
          {historial.map((c, i) => (
            <li key={i} className="space-y-3">
              <BurbujaUsuario texto={c.pregunta} />
              <BurbujaAsistente>
                {c.error ? (
                  <p className="text-red-600">{c.error}</p>
                ) : (
                  <>
                    <p className="whitespace-pre-wrap">{c.respuesta}</p>
                    {c.pasos!.length > 0 && <Pasos pasos={c.pasos!} />}
                  </>
                )}
              </BurbujaAsistente>
            </li>
          ))}
          {pensando && (
            <li className="space-y-3">
              <BurbujaUsuario texto={pensando} />
              <BurbujaAsistente>
                <Puntos />
                <span className="sr-only">El asistente está pensando</span>
              </BurbujaAsistente>
            </li>
          )}
        </ul>
        <div ref={finRef} />
        <form
          className="flex items-center gap-2 border-t border-gray-100 p-3"
          onSubmit={(e) => {
            e.preventDefault();
            enviar(pregunta);
          }}
        >
          <input
            value={pregunta}
            onChange={(e) => setPregunta(e.target.value)}
            placeholder="Escribe tu pregunta…"
            aria-label="Pregunta para el asistente"
            maxLength={500}
            className="flex-1 min-w-0 rounded-xl bg-gray-50 px-4 py-3 text-sm placeholder:text-gray-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-600"
          />
          <Button
            type="submit"
            size="icon"
            aria-label="Preguntar"
            disabled={!!pensando || !pregunta.trim()}
            className="h-11 w-11 rounded-xl"
          >
            <Send aria-hidden />
          </Button>
        </form>
      </div>
    </div>
  );
}

function BurbujaUsuario({ texto }: { texto: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-sm text-white">{texto}</p>
    </div>
  );
}

function BurbujaAsistente({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-50 text-blue-600">
        <Bot className="w-4 h-4" aria-hidden />
      </span>
      <div className="max-w-[85%] min-w-0 space-y-2 rounded-2xl rounded-tl-md bg-gray-100 px-4 py-2.5 text-sm text-gray-800">
        {children}
      </div>
    </div>
  );
}

function Pasos({ pasos }: { pasos: Paso[] }) {
  return (
    <details className="text-xs text-gray-600">
      <summary className="cursor-pointer select-none text-gray-500 hover:text-gray-700">
        <Wrench className="inline w-3 h-3 mr-1" aria-hidden />
        {pasos.length === 1 ? "1 herramienta usada" : `${pasos.length} herramientas usadas`}
      </summary>
      <ol className="mt-2 space-y-2 list-decimal pl-5">
        {pasos.map((p, j) => (
          <li key={j}>
            <code className="break-all">
              {p.herramienta}({JSON.stringify(p.argumentos)})
            </code>
            <pre className="mt-1 whitespace-pre-wrap break-all rounded-lg border border-gray-200 bg-white p-2">
              {JSON.stringify(p.resultado, null, 1)}
            </pre>
          </li>
        ))}
      </ol>
    </details>
  );
}
