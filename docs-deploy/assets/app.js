(function () {
  const search = document.querySelector("[data-search]");
  const searchable = Array.from(document.querySelectorAll("[data-searchable]"));
  const navLinks = Array.from(document.querySelectorAll(".sidebar a"));
  const mobileToggle = document.querySelector("[data-mobile-toggle]");
  const sidebar = document.querySelector(".sidebar");
  const searchToggle = document.querySelector("[data-search-toggle]");
  const searchPanel = document.querySelector("[data-search-panel]");
  const menuToggles = Array.from(document.querySelectorAll("[data-menu-toggle]"));
  const themeChoices = Array.from(document.querySelectorAll("[data-theme-choice]"));

  function closeMenus(exceptId) {
    menuToggles.forEach(function (toggle) {
      const id = toggle.getAttribute("data-menu-toggle");
      const menu = id ? document.getElementById(id) : null;
      if (!menu || id === exceptId) return;
      menu.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
    });
  }

  if (search) {
    search.addEventListener("input", function () {
      const query = search.value.trim().toLowerCase();
      searchable.forEach(function (node) {
        const text = node.textContent.toLowerCase();
        node.hidden = Boolean(query) && !text.includes(query);
      });
    });
  }

  if (searchToggle && searchPanel) {
    searchToggle.addEventListener("click", function () {
      const willOpen = searchPanel.hidden;
      searchPanel.hidden = !willOpen;
      searchToggle.setAttribute("aria-expanded", String(willOpen));
      closeMenus();
      if (willOpen && search) search.focus();
    });
  }

  menuToggles.forEach(function (toggle) {
    toggle.addEventListener("click", function () {
      const id = toggle.getAttribute("data-menu-toggle");
      const menu = id ? document.getElementById(id) : null;
      if (!menu) return;
      const willOpen = menu.hidden;
      closeMenus(willOpen ? id : undefined);
      menu.hidden = !willOpen;
      toggle.setAttribute("aria-expanded", String(willOpen));
      if (searchPanel && !searchPanel.hidden) {
        searchPanel.hidden = true;
        if (searchToggle) searchToggle.setAttribute("aria-expanded", "false");
      }
    });
  });

  themeChoices.forEach(function (choice) {
    choice.addEventListener("click", function () {
      const theme = choice.getAttribute("data-theme-choice") || "default";
      if (theme === "default") {
        document.documentElement.removeAttribute("data-theme");
        localStorage.removeItem("msrag-docs-theme");
      } else {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("msrag-docs-theme", theme);
      }
      closeMenus();
    });
  });

  const savedTheme = localStorage.getItem("msrag-docs-theme");
  if (savedTheme) {
    document.documentElement.setAttribute("data-theme", savedTheme);
  }

  document.addEventListener("click", function (event) {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (!target.closest(".tool-menu")) closeMenus();
    if (!target.closest(".search-panel") && !target.closest("[data-search-toggle]") && searchPanel && !searchPanel.hidden) {
      searchPanel.hidden = true;
      if (searchToggle) searchToggle.setAttribute("aria-expanded", "false");
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;
    closeMenus();
    if (searchPanel) searchPanel.hidden = true;
    if (searchToggle) searchToggle.setAttribute("aria-expanded", "false");
  });

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", function () {
      sidebar.classList.toggle("is-open");
    });
  }

  const sections = navLinks
    .map(function (link) {
      const id = link.getAttribute("href");
      return id && id.startsWith("#") ? document.querySelector(id) : null;
    })
    .filter(Boolean);

  const observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        navLinks.forEach(function (link) {
          link.classList.toggle("is-active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    },
    { rootMargin: "-30% 0px -55% 0px", threshold: 0.01 }
  );

  sections.forEach(function (section) {
    observer.observe(section);
  });

  document.querySelectorAll("a[href^='#']").forEach(function (link) {
    link.addEventListener("click", function () {
      if (sidebar) sidebar.classList.remove("is-open");
    });
  });
})();
