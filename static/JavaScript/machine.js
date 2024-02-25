initializeDropdowns("machine-dropdown");
initializeDropdowns("option-dropdown");

function initializeDropdowns(className) {
  var dropdowns = document.getElementsByClassName(className);
  var i, j, l, ll, selElmnt, a, b, c;

  for (i = 0; i < dropdowns.length; i++) {
    selElmnt = dropdowns[i].getElementsByTagName("select")[0];
    ll = selElmnt.length;

    a = document.createElement("DIV");
    a.setAttribute("class", "select-selected");
    a.innerHTML = selElmnt.options[selElmnt.selectedIndex].innerHTML;
    dropdowns[i].appendChild(a);

    b = document.createElement("DIV");
    b.setAttribute("class", "select-items select-hide");

    for (j = 1; j < ll; j++) {
      c = document.createElement("DIV");
      c.innerHTML = selElmnt.options[j].innerHTML;
      c.addEventListener("click", function (e) {
        var y, k, s, h, sl, yl;
        s = this.parentNode.parentNode.getElementsByTagName("select")[0];
        sl = s.length;
        h = this.parentNode.previousSibling;

        for (k = 0; k < sl; k++) {
          if (s.options[k].innerHTML == this.innerHTML) {
            s.selectedIndex = k;
            h.innerHTML = this.innerHTML;
            y = this.parentNode.getElementsByClassName("same-as-selected");
            yl = y.length;

            for (k = 0; k < yl; k++) {
              y[k].removeAttribute("class");
            }

            this.setAttribute("class", "same-as-selected");
            break;
          }
        }

        h.click();
      });
      b.appendChild(c);
    }

    dropdowns[i].appendChild(b);

    a.addEventListener("click", function (e) {
      e.stopPropagation();
      closeAllSelect(this);
      this.nextSibling.classList.toggle("select-hide");
      this.classList.toggle("select-arrow-active");
    });
  }
}

function closeAllSelect(elmnt) {
  var dropdowns = document.getElementsByClassName("select-items");
  var selectedDropdowns = document.getElementsByClassName("select-selected");
  var i, xl, yl, arrNo = [];

  for (i = 0; i < selectedDropdowns.length; i++) {
    if (elmnt == selectedDropdowns[i]) {
      arrNo.push(i);
    } else {
      selectedDropdowns[i].classList.remove("select-arrow-active");
    }
  }

  for (i = 0; i < dropdowns.length; i++) {
    if (arrNo.indexOf(i)) {
      dropdowns[i].classList.add("select-hide");
    }
  }
}

document.addEventListener("click", closeAllSelect);