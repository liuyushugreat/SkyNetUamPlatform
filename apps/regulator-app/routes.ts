/**
 * Regulator app route declarations (placeholder).
 *
 * Note: regulator is planned to be independently deployable and should keep
 * dependencies isolated.
 */

export const REGULATOR_BASE_PATH = "/regulator";

export type RegulatorRoute = {
  path: string;
  title: string;
};

export const regulatorRoutes: RegulatorRoute[] = [
  { path: REGULATOR_BASE_PATH, title: "Regulator Home" },
];


