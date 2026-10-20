// Single-row navbar: the top-level sections as links beside the wordmark.
document.addEventListener("DOMContentLoaded", function () {
  var navItems = [
    { name: "Getting Started", href: "/getting-started/" },
    { name: "Documentation", href: "/documentation/" },
    { name: "About", href: "/about/" }
  ];

  var headerInner = document.querySelector(".md-header__inner");
  var title = document.querySelector(".md-header__title");
  if (!headerInner || !title) return;

  // Links are resolved against the site root, which is not "/" on project Pages.
  var base = document.querySelector('link[rel="canonical"]');
  var root = "/";
  if (base) {
    try { root = new URL(base.href).pathname.replace(/[^/]*\/?$/, ""); } catch (e) {}
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
