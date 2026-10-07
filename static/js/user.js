(function () {
  "use strict";

  /* ---------------------------------------------------------------
     Helpers
  --------------------------------------------------------------- */

  const MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"];

  function pad(n) { return String(n).padStart(2, "0"); }

  // Format a Date as YYYY-MM-DD (the format the backend/db expects)
  function toISODate(date) {
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
  }

  // Format a Date as "01 September 2026" for display
  function toDisplayDate(date) {
    return `${pad(date.getDate())} ${MONTH_NAMES[date.getMonth()]} ${date.getFullYear()}`;
  }

  /* ---------------------------------------------------------------
     Avatar initials
  --------------------------------------------------------------- */

  function setAvatarInitials() {
    const name = (document.body.dataset.studentName || "").trim();
    if (!name) return;
    const parts = name.split(/\s+/).filter(Boolean);
    let initials = parts[0][0];
    if (parts.length > 1) initials += parts[1][0];
    document.getElementById("avatarInitials").textContent = initials.toUpperCase();
  }

  /* ---------------------------------------------------------------
     Calendar
  --------------------------------------------------------------- */

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  let viewYear = today.getFullYear();
  let viewMonth = today.getMonth(); // 0-indexed
  let selectedDate = new Date(today);

  const calendarTitle = document.getElementById("calendarTitle");
  const calendarGrid = document.getElementById("calendarGrid");
  const todayLabel = document.getElementById("todayLabel");

  // Convert JS getDay() (0=Sun) to a Mon-first column index (0=Mon..6=Sun)
  function mondayFirstIndex(jsDay) {
    return (jsDay + 6) % 7;
  }

  function renderCalendar() {
    calendarTitle.textContent = `${MONTH_NAMES[viewMonth]} ${viewYear}`;
    calendarGrid.innerHTML = "";

    const firstOfMonth = new Date(viewYear, viewMonth, 1);
    const leadingBlanks = mondayFirstIndex(firstOfMonth.getDay());
    const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
    const daysInPrevMonth = new Date(viewYear, viewMonth, 0).getDate();

    const totalCells = Math.ceil((leadingBlanks + daysInMonth) / 7) * 7;

    for (let cell = 0; cell < totalCells; cell++) {
      const dayNum = cell - leadingBlanks + 1;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "calendar__day";

      let cellDate;
      if (dayNum < 1) {
        cellDate = new Date(viewYear, viewMonth - 1, daysInPrevMonth + dayNum);
        btn.disabled = true;
      } else if (dayNum > daysInMonth) {
        cellDate = new Date(viewYear, viewMonth + 1, dayNum - daysInMonth);
        btn.disabled = true;
      } else {
        cellDate = new Date(viewYear, viewMonth, dayNum);
      }

      btn.textContent = cellDate.getDate();

      if (cellDate.getTime() === today.getTime()) btn.classList.add("is-today");
      if (cellDate.getTime() === selectedDate.getTime()) btn.classList.add("is-selected");

      if (!btn.disabled) {
        btn.addEventListener("click", () => {
          selectedDate = cellDate;
          renderCalendar();
          loadTimetable(selectedDate);
        });
      }

      calendarGrid.appendChild(btn);
    }
  }

  document.getElementById("prevMonth").addEventListener("click", () => {
    viewMonth -= 1;
    if (viewMonth < 0) { viewMonth = 11; viewYear -= 1; }
    renderCalendar();
  });

  document.getElementById("nextMonth").addEventListener("click", () => {
    viewMonth += 1;
    if (viewMonth > 11) { viewMonth = 0; viewYear += 1; }
    renderCalendar();
  });

  /* ---------------------------------------------------------------
     Timetable fetching
  --------------------------------------------------------------- */

  const timetableBody = document.getElementById("timetableBody");
  const timetableHeading = document.getElementById("timetableHeading");

  function renderTimetableState(message, isError) {
    timetableBody.innerHTML = "";
    const p = document.createElement("p");
    p.className = "timetable__state" + (isError ? " timetable__state--error" : "");
    p.textContent = message;
    timetableBody.appendChild(p);
  }

  function renderTimetableItems(items) {
    timetableBody.innerHTML = "";
    if (!items.length) {
      renderTimetableState("No timetable available for this date.", false);
      return;
    }
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "timetable__item";
      row.innerHTML = `
        <span class="timetable__time">${item.time || ""}</span>
        <span class="timetable__subject">${item.subject || ""}</span>
        <span class="timetable__faculty">${item.faculty || ""}</span>
      `;
      timetableBody.appendChild(row);
    });
  }

  async function loadTimetable(date) {
    const iso = toISODate(date);
    timetableHeading.textContent = `Timetable — ${toDisplayDate(date)}`;
    renderTimetableState("Loading timetable…", false);

    try {
      const res = await fetch(`/api/timetable?date=${encodeURIComponent(iso)}`);
      if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
      const data = await res.json();
      renderTimetableItems(Array.isArray(data) ? data : []);
    } catch (err) {
      renderTimetableState("Unable to load timetable. Please try again.", true);
    }
  }

  /* ---------------------------------------------------------------
     Sidebar panel switching (Profile / Attendance)
  --------------------------------------------------------------- */

  const sidebarLinks = document.querySelectorAll(".sidebar__link[data-target]");
  const panels = {
    "panel-attendance": document.getElementById("panel-attendance"),
    "panel-profile": document.getElementById("panel-profile"),
  };

  sidebarLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const target = link.dataset.target;

      sidebarLinks.forEach((l) => l.classList.toggle("is-active", l === link));
      Object.entries(panels).forEach(([key, el]) => {
        el.classList.toggle("panel--hidden", key !== target);
      });

      closeMobileMenu();
    });
  });

  /* ---------------------------------------------------------------
     Mobile off-canvas menu
  --------------------------------------------------------------- */

  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("mobileOverlay");
  const menuToggle = document.getElementById("menuToggle");

  function openMobileMenu() {
    sidebar.classList.add("is-open");
    overlay.classList.add("is-visible");
    menuToggle.setAttribute("aria-expanded", "true");
  }
  function closeMobileMenu() {
    sidebar.classList.remove("is-open");
    overlay.classList.remove("is-visible");
    menuToggle.setAttribute("aria-expanded", "false");
  }

  menuToggle.addEventListener("click", () => {
    sidebar.classList.contains("is-open") ? closeMobileMenu() : openMobileMenu();
  });
  overlay.addEventListener("click", closeMobileMenu);

  /* ---------------------------------------------------------------
     Init
  --------------------------------------------------------------- */

  setAvatarInitials();
  todayLabel.textContent = toDisplayDate(today);
  renderCalendar();
  loadTimetable(selectedDate);
})();
