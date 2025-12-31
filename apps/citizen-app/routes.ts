/**
 * Citizen app route declarations (placeholder).
 *
 * Current implementation uses App.tsx role switching (no react-router yet).
 * This file exists so we can later migrate to route aggregation without
 * reorganizing the app again.
 */

export const CITIZEN_BASE_PATH = "/citizen";

export type CitizenRoute = {
  path: string;
  title: string;
  // future: element, children, requiredScopes, etc.
};

export const citizenRoutes: CitizenRoute[] = [
  { path: CITIZEN_BASE_PATH, title: "Citizen Home" },
];


