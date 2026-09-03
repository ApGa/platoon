/*
 * Self-hosted Mermaid initialization.
 *
 * Material for MkDocs renders ```mermaid fences by lazily importing Mermaid from a public CDN at
 * runtime. That fails behind a corporate proxy or offline, and it leaves the site's diagrams
 * dependent on a third party. Platoon ships Mermaid locally instead, so the fences are emitted with
 * class "pl-mermaid" (see mkdocs.yml) — a class the theme ignores — and rendered here.
 */
(function () {
  "use strict";

  var SELECTOR = "pre.pl-mermaid";

  function themeName() {
    var scheme = document.body.getAttribute("data-md-color-scheme");
    if (!scheme) {
      scheme = document.documentElement.getAttribute("data-md-color-scheme");
    }
    return scheme === "slate" ? "dark" : "default";
  }

  function configure() {
    window.mermaid.initialize({
      startOnLoad: false,
      theme: themeName(),
      securityLevel: "strict",
      // Match the page font so label metrics are measured against what actually renders;
      // a mismatch here is the usual cause of clipped text.
      fontFamily: '"Inter", system-ui, sans-serif',
      fontSize: 13,
      flowchart: {
        curve: "basis",
        // Render at natural size and let the container scroll. useMaxWidth shrinks a wide
        // diagram to the content column, which is what made labels illegible.
        useMaxWidth: false,
        htmlLabels: true,
        nodeSpacing: 34,
        rankSpacing: 44,
        padding: 10,
        diagramPadding: 8,
      },
      sequence: {
        useMaxWidth: false,
        wrap: true,
        width: 168,
        boxMargin: 12,
        messageFontSize: 13,
        actorFontSize: 13,
        noteFontSize: 12,
      },
    });
  }

  var counter = 0;

  function render(root) {
    if (!window.mermaid) return;
    var blocks = (root || document).querySelectorAll(SELECTOR);
    if (!blocks.length) return;
    configure();
    blocks.forEach(function (block) {
      // Stash the source on first sight: rendering replaces the element's content, and a theme
      // switch has to re-render from the original text.
      if (!block.dataset.plSource) {
        block.dataset.plSource = (block.textContent || "").trim();
      }
      var source = block.dataset.plSource;
      if (!source) return;
      var id = "pl-mermaid-" + counter++;
      window.mermaid
        .render(id, source)
        .then(function (result) {
          block.innerHTML = result.svg;
          block.classList.add("pl-mermaid--rendered");
          if (result.bindFunctions) {
            result.bindFunctions(block);
          }
        })
        .catch(function (error) {
          // Leave the source visible rather than showing an empty box.
          block.classList.add("pl-mermaid--failed");
          block.textContent = source;
          if (window.console && console.warn) {
            console.warn("Mermaid failed to render a diagram:", error);
          }
        });
    });
  }

  function rerenderAll() {
    document.querySelectorAll(SELECTOR).forEach(function (block) {
      block.classList.remove("pl-mermaid--rendered", "pl-mermaid--failed");
    });
    render(document);
  }

  function start() {
    render(document);

    // Re-render when the reader toggles light/dark so diagram colors follow the page.
    var observer = new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
        if (mutations[i].attributeName === "data-md-color-scheme") {
          rerenderAll();
          return;
        }
      }
    });
    observer.observe(document.body, { attributes: true });
  }

  // With navigation.instant enabled, Material replaces the document body on every page change and
  // publishes the new document on the document$ observable.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () {
      render(document);
    });
    if (document.readyState !== "loading") {
      start();
    } else {
      document.addEventListener("DOMContentLoaded", start);
    }
  } else if (document.readyState !== "loading") {
    start();
  } else {
    document.addEventListener("DOMContentLoaded", start);
  }
})();
