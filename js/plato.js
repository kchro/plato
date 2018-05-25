/**
 *
 */
function find(root, selector) {
  return $(root).find(selector)[0];
}

/**
 *
 */
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

/**
 *
 */
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

/**
 *
 */
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

function createCORSRequest(method, url){
    var xhr = new XMLHttpRequest();
    if ("withCredentials" in xhr){
        xhr.open(method, url, true);
    } else if (typeof XDomainRequest != "undefined"){
        xhr = new XDomainRequest();
        xhr.open(method, url);
    } else {
        xhr = null;
    }
    return xhr;
}

/**
 * post
 */
function post(fol, params, callback) {
  let formData = new FormData();
  formData.append("sig", fol);
  formData.append("rules", params.rules);
  formData.append("rulefile", params.rulefile);
  formData.append("blockrules", params.blockrules);
  formData.append("blockfile", params.blockfile);

  let request = createCORSRequest('POST', 'http://cypriot.stanford.edu:8080/ace/');
  request.onreadystatechange = function() {
    if (request.readyState == 4 && request.status == 200) {
      callback(request);
    } else {
      console.log(request);
    }
  }

  console.log(formData);
  request.send(formData);
}

function renderErrors(root, state) {
  if (state.error_log[state.problem] === undefined) {
    state.error_log[state.problem] = [];
  }
  const children = [];
  state.error_log[state.problem].forEach(function(error) {
    const child = {
      'element': 'div',
      'innerHTML': error
    }
    children.push(child);
  });
  const errorLog = renderObject({
    'element': 'div',
    'className': 'plato-answer-errorlog',
    'children': children
  });
  return errorLog;
}

/**
 *
 */
function renderTrans(root, state) {
  // const translation = document.createElement("div");
  // translation.className = "plato-translation";
  const translation = renderObject({
    'element': 'div',
    'className': 'plato-translation'
  });

  const prompt = renderObject({
    'element': 'div',
    'className': 'plato-prompt',
    'children': [
      {
        'element': 'div',
        'className': 'plato-prompt-label',
        'innerHTML': 'English'
      }
    ]
  });

  const promptText = renderObject({
    'element': 'div',
    'className': 'plato-prompt-text'
  });

  // promptText.innerHTML = data.problems[state.problem].prompt;
  post(data.problems[state.problem].answer, data.config, function(response) {
    // get the list of NL sentences and choose based on policy
    const prompts = response.responseText.split('\n');

    // random policy
    const index = Math.floor(Math.random() * prompts.length);

    promptText.innerHTML = prompts[index];
  });

  prompt.appendChild(promptText);

  const answer = renderObject({
    'element': 'div',
    'className': 'plato-answer',
    'children': [
      {
        'element': 'div',
        'className': 'plato-answer-label',
        'innerHTML': 'First-Order Logic'
      },
      renderErrors(root, state),
      {
        'element': 'input',
        'className': 'plato-answer-text',
        'placeholder': 'Enter FOL Translation'
      }
    ]
  })

  const submit = renderObject({
    'element': 'button',
    'type': 'button',
    'className': 'btn btn-primary plato-submit',
    'innerHTML': 'submit'
  });

  let alert;
  $(submit).on("click", function() {
    if (alert) {
      translation.removeChild(alert);
    }

    const submission = $('.plato-answer-text').val();
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

      state.error_log[state.problem].push(submission);

      render(root, state);
      alert = displayFeedback(translation, state);
    }
  });

  // compile
  translation.appendChild(prompt);
  translation.appendChild(answer);
  translation.appendChild(submit);
  return translation;
}

/**
 *
 */
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

  // compile
  progress.appendChild(progbar);
  return progress;
}

/**
 *
 */
function renderApp(root, state) {
  const app = renderObject({
    'element': 'div',
    'className': 'plato-app',
    'children': [
      {
        'element': 'div',
        'className': 'plato-problem',
        'children': [
          {
            'element': 'div',
            'className': 'plato-problem-header',
            'innerHTML': data.problems[state.problem].label
          },
          renderTrans(root, state)
        ]
      },
      renderProgress(state)
    ]
  });

  return app;
}

/**
 *
 */
function render(root, state) {
  let app, body;

  if (state.init == false) {
    /*
     * render the application structure
     */
    app = renderApp(root, state);
    body = $(root).find(".card-body")[0];
    body.appendChild(app);

    // structure is built
    state.init = true;

  } else if (state.problem >= data.problems.length) {
    body = $(root).find(".card-body")[0];
    $(body).empty();

    const analytics = renderObject({
      'element': 'div',
      'className': 'plato-analytics-wrapper',
      'children': [
        {
          'element': 'div',
          'className': 'plato-analytics',
          'children': [
            {
              'element': 'div',
              'className': 'plato-analytics-header',
              'innerHTML': 'Results'
            }
          ]
        },
        renderProgress(state)
      ]
    });

    // const wrapper = document.createElement("div");
    // wrapper.className = "plato-analytics-wrapper";
    // const analytics = document.createElement("div");
    // analytics.className = "plato-analytics";
    // const header = document.createElement("div");
    // header.className = "plato-analytics-header";
    // header.innerHTML = "Results";
    // analytics.appendChild(header);
    // wrapper.appendChild(analytics);
    // body.appendChild(wrapper);

    body.appendChild(analytics);

    displayAnalytics(analytics, state);

    // const progress = renderProgress(state);
    // wrapper.appendChild(progress);
  } else {
    body = $(root).find(".card-body")[0];
    $(body).empty();

    app = renderApp(root, state);
    body.appendChild(app);
  }
  console.log(root);
}

/**
 *
 */
$(function() {
  let root = document.getElementById("plato-app");

  const state = {
    "init": false,
    "problem": 2,
    "random": true,
    "analytics": [
      {
        "label":"correct",
        "value":0
      },
      {
        "label":"incorrect",
        "value":0
      }
    ],
    "error_log": {}
  };

  render(root, state);
});
