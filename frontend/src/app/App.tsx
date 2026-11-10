import { useRef, useState, type ChangeEvent } from "react";
import { Building2, ImageUp, Loader2, Play, Trash2 } from "lucide-react";
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

async function analizar(archivo: File): Promise<Resultado> {
  const form = new FormData();
  form.append("foto", archivo);
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
      const r = await analizar(foto.archivo);
      setFotos((prev) => ({ ...prev, [salonId]: { ...foto, resultado: r, visibles: r.detecciones.map(() => true) } }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAnalizando(false);
      estadoRef.current?.focus();
    }
  };

  const quitar = () => {
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

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-5xl mx-auto px-4 h-16 flex items-center gap-3">
          <Building2 className="w-8 h-8 text-blue-600" aria-hidden />
          <div>
            <h1 className="text-xl font-bold text-gray-900">Conteo de personas por rol</h1>
            <p className="text-sm text-gray-500">Universidad EAFIT</p>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8 space-y-6">
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
              <p className="font-semibold text-gray-900">{s.nombre}</p>
              <p className="text-sm text-gray-500">
                {fotos[s.id]?.resultado
                  ? `${fotos[s.id].resultado!.total} de ${s.capacidad} personas`
                  : fotos[s.id]
                    ? "Foto sin procesar"
                    : `Capacidad: ${s.capacidad}`}
              </p>
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
