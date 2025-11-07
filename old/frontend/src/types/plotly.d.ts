declare module 'plotly.js-dist-min' {
  import { Layout } from 'plotly.js';
  
  export function newPlot(
    root: HTMLElement,
    data: any[],
    layout?: Partial<Layout>,
    config?: any
  ): void;
  
  export function purge(root: HTMLElement): void;
} 