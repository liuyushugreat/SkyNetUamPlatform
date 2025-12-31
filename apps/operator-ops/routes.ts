/**
 * Operator Ops route declarations (placeholder).
 *
 * Current implementation uses App.tsx role switching (no react-router yet).
 */

export const OPERATOR_BASE_PATH = "/operator";

export type OperatorRoute = {
  path: string;
  title: string;
};

export const operatorRoutes: OperatorRoute[] = [
  { path: OPERATOR_BASE_PATH, title: "Operator Ops Home" },
];


