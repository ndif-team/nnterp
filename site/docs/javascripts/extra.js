// Single-row navbar: the top-level sections as links beside the wordmark.
document.addEventListener("DOMContentLoaded", function () {
  // The encyclopedia banner, above the header on the home page.
  if (document.querySelector(".nn-hero")) {
    var banner = document.createElement("div");
    banner.className = "nn-announcement";
    banner.innerHTML = 'Every family\u2019s block, drawn with nnterp\u2019s names: ' +
      '<a href="encyclopedia/" target="_blank" rel="noopener"><u>the nnterp encyclopedia</u></a> \u2192';
    document.body.insertBefore(banner, document.body.firstChild);
  }

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
  var title = document.querySelector(".md-header__button.md-logo");
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

  // After the repository link: the nnsight logo (nnsight.net) and the encyclopedia's
  // mark (the encyclopedia, in a new tab).
  var source = document.querySelector(".md-header__source");
  if (source) {
    var links = document.createElement("div");
    links.className = "nn-header-links";
    var nnsight = document.createElement("a");
    nnsight.href = "https://nnsight.net";
    nnsight.title = "nnsight";
    var nnsightImg = document.createElement("img");
    nnsightImg.src = root.replace(/\/$/, "") + "/assets/nnsight.svg";
    nnsightImg.alt = "nnsight";
    nnsight.appendChild(nnsightImg);
    var enc = document.createElement("a");
    enc.href = root.replace(/\/$/, "") + "/encyclopedia/";
    enc.title = "The nnterp encyclopedia";
    enc.target = "_blank";
    enc.rel = "noopener";
    var encImg = document.createElement("img");
    encImg.src = root.replace(/\/$/, "") + "/assets/encyclopedia.svg";
    encImg.alt = "encyclopedia";
    enc.appendChild(encImg);
    links.appendChild(nnsight);
    links.appendChild(enc);
    source.parentNode.insertBefore(links, source.nextSibling);
  }
});

// "Copy page": fetch the page's Markdown and put it on the clipboard.
document.addEventListener("click", function (event) {
  var button = event.target.closest(".nn-copy-md");
  if (!button) return;
  fetch(button.getAttribute("data-markdown"))
    .then(function (r) { return r.text(); })
    .then(function (text) { return navigator.clipboard.writeText(text); })
    .then(function () {
      var label = button.textContent;
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = label; }, 1500);
    })
    .catch(function () { button.textContent = "Copy failed"; });
});
