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

interface Resultado {
  total: number;
  roles: { rol: string; color: string; personas: number }[];
  imagen: string;
}

interface Foto {
  archivo: File;
  url: string;
  resultado?: Resultado;
}

async function analizar(archivo: File): Promise<Resultado> {
  const form = new FormData();
  form.append("foto", archivo);
  let respuesta: Response;
  try {
    respuesta = await fetch("/analizar", { method: "POST", body: form });
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
    setError("");
    setFotos((prev) => ({ ...prev, [salonId]: { archivo, url: URL.createObjectURL(archivo) } }));
    estadoRef.current?.focus();
  };

  const procesar = async () => {
    setAnalizando(true);
    setError("");
    try {
      const r = await analizar(foto.archivo);
      setFotos((prev) => ({ ...prev, [salonId]: { ...foto, resultado: r } }));
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
            accept="image/jpeg,image/png,image/webp"
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

        {foto && (
          <div className={cn("grid grid-cols-1 gap-6 items-start", resultado && "lg:grid-cols-[1fr_16rem]")}>
            <img
              src={resultado?.imagen ?? foto.url}
              alt={
                resultado
                  ? `Foto de ${salon.nombre} con ${resultado.total} personas remarcadas por rol`
                  : `Foto de ${salon.nombre} sin procesar`
              }
              className="w-full rounded-xl border border-gray-200"
            />
            {resultado && (
              <Card>
                <CardContent className="p-6 space-y-4">
                  <div>
                    <p className="text-sm text-gray-500">Personas</p>
                    <p className="text-4xl font-bold text-gray-900">{resultado.total}</p>
                    <p className="text-sm text-gray-500">
                      {Math.round((resultado.total / salon.capacidad) * 100)}% de la capacidad
                    </p>
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
            )}
          </div>
        )}
      </main>
    </div>
  );
}
