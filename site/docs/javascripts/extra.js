// Single-row navbar: the top-level sections as links beside the wordmark.
document.addEventListener("DOMContentLoaded", function () {
  var navItems = [
    { name: "Getting Started", href: "/getting-started/" },
    { name: "Guide", href: "/guide/" },
    { name: "Tutorials", href: "/tutorials/" },
    { name: "vLLM", href: "/vllm/" },
    { name: "Developing", href: "/developing/" },
    { name: "Documentation", href: "/documentation/" },
    { name: "About", href: "/about/" }
  ];

  var headerInner = document.querySelector(".md-header__inner");
  var title = document.querySelector(".md-header__title");
  if (!headerInner || !title) return;

  // Links are resolved against the site root, which is not "/" on project Pages. Material's
  // (hidden) home link points there from every page.
  var home = document.querySelector(".md-header__button.md-logo");
  var root = "/";
  if (home) {
    try { root = new URL(home.href).pathname; } catch (e) {}
  }
  var current = window.location.pathname;

  var headerNav = document.createElement("nav");
  headerNav.className = "header-nav";
  navItems.forEach(function (item) {
    var link = document.createElement("a");
    var href = root.replace(/\/$/, "") + item.href;
    link.href = href;
    link.textContent = item.name;
    if (current === href || current.indexOf(href) === 0) link.classList.add("active");
    headerNav.appendChild(link);
  });
  title.parentNode.insertBefore(headerNav, title.nextSibling);
});
