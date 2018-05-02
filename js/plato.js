function find(root, selector) {
  return $(root).find(selector)[0];
}

function displayCorrect(translation, state) {
  const alert = document.createElement("div");
  alert.className = "alert alert-success";
  alert.innerHTML = "Correct!";
  translation.appendChild(alert);

  $(alert).click(function() {
    translation.removeChild(alert);
  });

  return alert;
}

function displayFeedback(translation, state) {
  const alert = document.createElement("div");
  alert.className = "alert alert-danger";
  alert.innerHTML = "Incorrect!";
  translation.appendChild(alert);

  $(alert).click(function() {
    translation.removeChild(alert);
  });

  return alert;
}

function displayAnalytics(body, state) {
  const w = 300,                            //width
        h = 300,                            //height
        r = 150,                            //radius
        color = d3.scaleOrdinal()
          .range(['#5cb85c', '#d9534f']);     //builtin range of colors

  const analytics = state.analytics;

  var vis = d3.select(body)
      .append("svg")
      .data([analytics])                    //associate our data with the document
        .attr("width", w)                 //set the width and height of our visualization (these will be attributes of the <svg> tag
        .attr("height", h)
      .append("g")                        //make a group to hold our pie chart
        .attr("transform", "translate(" + r + "," + r + ")")    //move the center of the pie chart from 0, 0 to radius, radius

  var arc = d3.arc()              //this will create <path> elements for us using arc data
              .outerRadius(r)
              .innerRadius(r-10);

  var pie = d3.pie()           //this will create arc data for us given a list of values
              .value(function(d) { return d.value; });    //we must tell it out to access the value of each element in our data array

  var arcs = vis.selectAll("g.slice")     //this selects all <g> elements with class slice (there aren't any yet)
    .data(pie)                          //associate the generated pie data (an array of arcs, each having startAngle, endAngle and value properties)
    .enter()                            //this will create <g> elements for every "extra" data element that should be associated with a selection. The result is creating a <g> for every object in the data array
      .append("g")                      //create a group to hold each slice (we will have a <path> and a <text> element associated with each slice)
        .attr("class", "slice");        //allow us to style things in the slices (like text)

  arcs.append("path")
    .attr("fill", function(d, i) { return color(i); } ) //set the color for each slice to be chosen from the color function defined above
    .attr("d", arc);                                    //this creates the actual SVG path using the associated data (pie) with the arc drawing function

  const total = analytics[0].value + analytics[1].value;
  const rate = ((analytics[0].value / total) * 100).toFixed(2);

  arcs.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", -10)
    .text(rate + "% acceptance rate");

  arcs.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", 10)
    .text(analytics[0].value + " correct out of " + total + " attempts");

}

function renderTrans(root, state) {
  const translation = document.createElement("div");
  translation.className = "plato-translation";

  const prompt = document.createElement("div");
  prompt.className = "plato-prompt";
  const promptLabel = document.createElement("div");
  promptLabel.className = "plato-prompt-label";
  promptLabel.innerHTML = "English";
  const promptText = document.createElement("div");
  promptText.className = "plato-prompt-text";
  promptText.innerHTML = data.problems[state.problem].prompt;
  prompt.appendChild(promptLabel);
  prompt.appendChild(promptText);

  const answer = document.createElement("div");
  answer.className = "plato-answer";
  const answerLabel = document.createElement("div");
  answerLabel.className = "plato-answer-label";
  answerLabel.innerHTML = "First-Order Logic";
  const answerText = document.createElement("input");
  answerText.placeholder = "Enter FOL Translation";
  answerText.className = "plato-answer-text";
  answer.appendChild(answerLabel);
  answer.appendChild(answerText);

  const submit = document.createElement("button");
  submit.type = "button";
  submit.className = "btn btn-primary plato-submit";
  submit.innerHTML = "submit";

  let alert;
  $(submit).on("click", function() {
    if (alert) {
      translation.removeChild(alert);
    }

    const submission = $(answerText).val();
    if (submission == data.problems[state.problem].answer) {
      state.analytics[0].value++;

      alert = displayCorrect(translation, state);
      submit.innerHTML = "next";
      submit.className = "btn btn-success plato-submit";
      $(submit).off("click");
      $(submit).on("click", function() {
        state.problem++;
        render(root, state);
      });

    } else {
      state.analytics[1].value++;

      alert = displayFeedback(translation, state);
    }
  });

  translation.appendChild(prompt);
  translation.appendChild(answer);
  translation.appendChild(submit);
  return translation;
}

function renderProgress(state) {
  const value = (state.problem * 100 / data.problems.length).toFixed(2);

  const progress = document.createElement("div");
  progress.className = "progress plato-progress";
  const progbar = document.createElement("div");
  progbar.className = "progress-bar";
  progbar.role = "progressbar";
  $(progbar).attr("aria-valuemin", 0);
  $(progbar).attr("aria-valuemax", 100.00);
  $(progbar).attr("aria-valuenow", value).css("width", value+"%");
  progress.appendChild(progbar);
  return progress;
}

function renderApp(root, state) {
  const app = document.createElement("div");
  app.className = "plato-app";
  const problem = document.createElement("div");
  problem.className = "plato-problem";
  const problemHeader = document.createElement("div");
  problemHeader.className = "plato-problem-header";
  problemHeader.innerHTML = data.problems[state.problem].label;
  const translation = renderTrans(root, state);
  problem.appendChild(problemHeader);
  problem.appendChild(translation);
  app.appendChild(problem);

  const progress = renderProgress(state);
  app.appendChild(progress);

  return app;
}

function render(root, state) {
  let app, body;
  if (state.init == false) {
    app = renderApp(root, state);
    body = $(root).find(".card-body")[0];
    body.appendChild(app);
    state.init = true;
  } else if (state.problem >= data.problems.length) {
    body = $(root).find(".card-body")[0];
    $(body).empty();

    const wrapper = document.createElement("div");
    wrapper.className = "plato-analytics-wrapper";
    const analytics = document.createElement("div");
    analytics.className = "plato-analytics";
    const header = document.createElement("div");
    header.className = "plato-analytics-header";
    header.innerHTML = "Results";
    analytics.appendChild(header);
    wrapper.appendChild(analytics);
    body.appendChild(wrapper);

    displayAnalytics(analytics, state);

    const progress = renderProgress(state);
    wrapper.appendChild(progress);
  } else {
    body = $(root).find(".card-body")[0];
    $(body).empty();

    app = renderApp(root, state);
    body.appendChild(app);
  }
  console.log(root);
}

$(function() {
  let root = document.getElementById("plato-app");

  const state = {
    "init": false,
    "problem": 0,
    "analytics": [
      {
        "label":"correct",
        "value":0
      },
      {
        "label":"incorrect",
        "value":0
      }
    ]
  };

  render(root, state);

});
