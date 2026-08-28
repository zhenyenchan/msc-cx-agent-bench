/**
 * Annotation program — collector (v2, 50 tasks).
 * Appends one row per tagger x task to the sheet named "responses".
 */

var HEADERS = [
  "tagger_name","task_id","difficulty","score","comments","flag_comment",
  "seconds_on_task","seconds_session_elapsed","sitting","submitted_at",
  "session_id","presentation_order","received_at"
];

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    var d = JSON.parse(e.postData.contents);
    var sh = getSheet();
    sh.appendRow(HEADERS.map(function (h) {
      return h === "received_at" ? new Date() : (d[h] !== undefined ? d[h] : "");
    }));
    return json({ ok: true });
  } catch (err) {
    return json({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return json({ ok: true, msg: "collector alive" });
}

function getSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sh = ss.getSheetByName("responses");
  if (!sh) { sh = ss.insertSheet("responses"); }
  if (sh.getLastRow() === 0) {
    sh.appendRow(HEADERS);
    sh.getRange(1, 1, 1, HEADERS.length).setFontWeight("bold");
    sh.setFrozenRows(1);
  }
  return sh;
}

/** Optional: run manually (Run > progressReport) to see how far each tagger has got. */
function progressReport() {
  var rows = getSheet().getDataRange().getValues();
  var byTagger = {};
  for (var i = 1; i < rows.length; i++) {
    var name = rows[i][0];
    byTagger[name] = (byTagger[name] || 0) + 1;
  }
  Logger.log(byTagger);
}

function json(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}