import { defineConfig } from "astro/config";
import svelte from "@astrojs/svelte";
// import postcss from "@astrojs/postcss";
// import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";

// https://astro.build/config
export default defineConfig({
  site: "https://jakecupani.github.io",
  integrations: [svelte(), mdx()],
  markdown: {
    shikiConfig: {
      theme: "nord",
    },
    rehypePlugins: [
      [
        "rehype-external-links",
        {
          target: "_blank",
        },
      ],
    ],
  },
});