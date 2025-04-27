import { defineConfig } from "astro/config";
import svelte from "@astrojs/svelte";
import preprocess from "svelte-preprocess";
import postcss from 'postcss';


export default {
  integrations: [
    // ...other integrations,
    postcss(),
  ],
};