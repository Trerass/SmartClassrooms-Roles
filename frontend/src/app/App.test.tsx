import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

const RESULTADO = {
  total: 2,
  roles: [{ rol: "Estudiante", color: "#000000", personas: 2 }],
  imagen: "data:image/jpeg;base64,AA",
};

const respuesta = (cuerpo: unknown, status = 200) => new Response(JSON.stringify(cuerpo), { status });
const boton = (nombre: string) => screen.getByRole("button", { name: nombre });
const botonOpcional = (nombre: string) => screen.queryByRole("button", { name: nombre });
const salon = (nombre: RegExp) => screen.getByRole("button", { name: nombre }) as HTMLButtonElement;
const estado = () => screen.getByRole("status").textContent;

function elegir(nombre = "a.jpg", tipo = "image/jpeg") {
  fireEvent.change(screen.getByLabelText(/^Foto de/), { target: { files: [new File(["x"], nombre, { type: tipo })] } });
}

let fetchFalso: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchFalso = vi.fn();
  vi.stubGlobal("fetch", fetchFalso);
  URL.createObjectURL = vi.fn(() => "blob:foto");
  URL.revokeObjectURL = vi.fn();
  render(<App />);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("subir una foto no la analiza hasta pulsar Procesar", () => {
  elegir();
  expect(fetchFalso).not.toHaveBeenCalled();
  expect(boton("Procesar")).toBeTruthy();
  expect(boton("Quitar foto")).toBeTruthy();
  expect(botonOpcional("Subir foto")).toBeNull();
  expect(screen.getByRole("img").getAttribute("src")).toBe("blob:foto");
});

it("procesar muestra la imagen anotada y el conteo", async () => {
  fetchFalso.mockResolvedValue(respuesta(RESULTADO));
  elegir();
  fireEvent.click(boton("Procesar"));
  expect(await screen.findByText("Estudiante")).toBeTruthy();
  expect(screen.getByRole("img").getAttribute("src")).toBe(RESULTADO.imagen);
  expect(fetchFalso.mock.calls[0][0]).toBe("/analizar");
  expect(botonOpcional("Procesar")).toBeNull();
});

it("quitar borra la foto y el resultado y libera la memoria", async () => {
  fetchFalso.mockResolvedValue(respuesta(RESULTADO));
  elegir();
  fireEvent.click(boton("Procesar"));
  await screen.findByText("Estudiante");
  fireEvent.click(boton("Quitar foto"));
  expect(screen.queryByRole("img")).toBeNull();
  expect(screen.queryByText("Estudiante")).toBeNull();
  expect(boton("Subir foto")).toBeTruthy();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:foto");
});

it("cada salón conserva su propia foto", () => {
  elegir();
  fireEvent.click(salon(/38-102/));
  expect(screen.queryByRole("img")).toBeNull();
  expect(boton("Subir foto")).toBeTruthy();
  fireEvent.click(salon(/38-101/));
  expect(screen.getByRole("img")).toBeTruthy();
});

it("los salones se bloquean durante el análisis y vuelven después", async () => {
  let responder!: (r: Response) => void;
  fetchFalso.mockReturnValue(new Promise((r) => (responder = r)));
  elegir();
  fireEvent.click(boton("Procesar"));
  expect(salon(/38-102/).disabled).toBe(true);
  expect((boton("Quitar foto") as HTMLButtonElement).disabled).toBe(true);
  responder(respuesta(RESULTADO));
  await screen.findByText("Estudiante");
  expect(salon(/38-102/).disabled).toBe(false);
});

it("muestra el mensaje de la API y permite reintentar", async () => {
  fetchFalso.mockResolvedValueOnce(respuesta({ detail: "El archivo no es una imagen válida." }, 400));
  elegir();
  fireEvent.click(boton("Procesar"));
  expect(await screen.findByText("El archivo no es una imagen válida.")).toBeTruthy();

  fetchFalso.mockResolvedValueOnce(respuesta(RESULTADO));
  fireEvent.click(boton("Procesar"));
  expect(await screen.findByText("Estudiante")).toBeTruthy();
  expect(estado()).not.toContain("válida");
});

it("avisa cuando la API está cargando o apagada", async () => {
  fetchFalso.mockResolvedValue(new Response("", { status: 500 }));
  elegir();
  fireEvent.click(boton("Procesar"));
  expect(await screen.findByText(/no está listo/)).toBeTruthy();
});

it("avisa cuando no hay conexión o se agota el tiempo", async () => {
  fetchFalso.mockRejectedValue(new TypeError("Failed to fetch"));
  elegir();
  fireEvent.click(boton("Procesar"));
  expect(await screen.findByText(/No se pudo conectar/)).toBeTruthy();
  expect(fetchFalso.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
});

it("rechaza formatos no admitidos sin subirlos", () => {
  elegir("foto.heic", "image/heic");
  expect(estado()).toContain("Formato no admitido");
  expect(screen.queryByRole("img")).toBeNull();
  expect(boton("Subir foto")).toBeTruthy();
});

it("cambiar de salón borra el error anterior", () => {
  elegir("foto.heic", "image/heic");
  fireEvent.click(salon(/38-102/));
  expect(estado()).not.toContain("Formato no admitido");
});
