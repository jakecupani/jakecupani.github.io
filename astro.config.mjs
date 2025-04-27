import { defineConfig } from 'astro/config';
import svelte from '@astrojs/svelte';
import postcss from '@astrojs/postcss';

export default defineConfig({
  integrations: [svelte(), postcss()],
});