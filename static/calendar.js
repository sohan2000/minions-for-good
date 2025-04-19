document.addEventListener("DOMContentLoaded", function () {
    const calendarContainer = document.getElementById("calendar-grid");
    const calendarTitle = document.getElementById("calendar-title");
    const prevButton = document.getElementById("prev-month");
    const nextButton = document.getElementById("next-month");

    const monthNames = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];

    let current = new Date();
    let currentMonth = current.getMonth();
    let currentYear = current.getFullYear();

    function renderCalendar(month, year) {
        calendarContainer.innerHTML = "";

        const today = new Date();
        const daysInMonth = new Date(year, month + 1, 0).getDate();
        const firstDayOfWeek = new Date(year, month, 1).getDay();

        calendarTitle.innerText = `${monthNames[month]} ${year}`;

        const blanks = firstDayOfWeek === 0 ? 6 : firstDayOfWeek - 1;
        for (let i = 0; i < blanks; i++) {
            const blankCell = document.createElement("div");
            blankCell.className = "calendar-day empty";
            calendarContainer.appendChild(blankCell);
        }

        for (let i = 1; i <= daysInMonth; i++) {
            const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(i).padStart(2, '0')}`;
            const dayCell = document.createElement("div");
            dayCell.className = "calendar-day";
            dayCell.innerText = i;

            if (i === today.getDate() && month === today.getMonth() && year === today.getFullYear()) {
                dayCell.classList.add("today");
            }

            if (highlightedDates.includes(dateStr)) {
                dayCell.classList.add("highlight");
            }

            calendarContainer.appendChild(dayCell);
        }
    }

    prevButton.addEventListener("click", () => {
        currentMonth = (currentMonth === 0) ? 11 : currentMonth - 1;
        currentYear = (currentMonth === 11) ? currentYear - 1 : currentYear;
        renderCalendar(currentMonth, currentYear);
    });

    nextButton.addEventListener("click", () => {
        currentMonth = (currentMonth === 11) ? 0 : currentMonth + 1;
        currentYear = (currentMonth === 0) ? currentYear + 1 : currentYear;
        renderCalendar(currentMonth, currentYear);
    });

    renderCalendar(currentMonth, currentYear);
});