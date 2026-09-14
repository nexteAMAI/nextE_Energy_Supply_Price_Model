// Render docs/USER_GUIDE.md as a Word document in the format of the legacy user guide
// (US Letter, 1" margins, title block, Contents, Heading 1/2, bullets, running header and
// page-numbered footer). Usage: node tools/build_user_guide_docx.js docs/USER_GUIDE.md out.docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Header, Footer, PageNumber,
  LevelFormat, PageBreak, BorderStyle,
} = require("docx");

const [src, out] = process.argv.slice(2);
const md = fs.readFileSync(src, "utf8").split(/\r?\n/);
const NAVY = "1F3E66";
const FONT = "Montserrat";

const inline = (text) => {
  // **bold**, `code` and plain runs
  const runs = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT }));
    const tok = m[0];
    if (tok.startsWith("**")) runs.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT }));
    else runs.push(new TextRun({ text: tok.slice(1, -1), font: "JetBrains Mono", size: 20 }));
    last = m.index + tok.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), font: FONT }));
  return runs;
};

const children = [];
let title = "", subtitle = "";
let toc = [];
let i = 0;
// title = first "# " line; the paragraph after it is the version line
for (; i < md.length; i++) {
  const l = md[i];
  if (l.startsWith("# ")) { title = l.slice(2).replace("USER GUIDE - ", ""); break; }
}
i++;
let para = [];
const flush = () => {
  if (para.length) {
    children.push(new Paragraph({ children: inline(para.join(" ")), spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED }));
    para = [];
  }
};
const body = [];
let first = true;
for (; i < md.length; i++) {
  const l = md[i];
  if (first && l.trim() && !l.startsWith("#")) { subtitle = l.trim(); first = false; continue; }
  if (l.startsWith("## ")) { flush(); toc.push(l.slice(3)); body.push({ h: 1, t: l.slice(3) }); continue; }
  if (l.startsWith("### ")) { flush(); body.push({ h: 2, t: l.slice(4) }); continue; }
  if (/^- /.test(l)) { flush(); body.push({ b: l.slice(2) }); continue; }
  if (/^\d+\. /.test(l)) { flush(); body.push({ n: l.replace(/^\d+\. /, "") }); continue; }
  if (/^\|/.test(l)) { continue; }
  if (!l.trim()) { flush(); continue; }
  if (para.length && (body.length && (body[body.length - 1].b !== undefined || body[body.length - 1].n !== undefined)) && /^\s{2,}/.test(l)) {
    // continuation of a list item
    const last = body[body.length - 1];
    if (last.b !== undefined) last.b += " " + l.trim(); else last.n += " " + l.trim();
    continue;
  }
  if (body.length && (body[body.length - 1].b !== undefined || body[body.length - 1].n !== undefined) && /^\s{2,}/.test(l)) {
    const last = body[body.length - 1];
    if (last.b !== undefined) last.b += " " + l.trim(); else last.n += " " + l.trim();
    continue;
  }
  para.push(l.trim());
  // paragraphs end at blank lines; but flush into body via marker
  if (i + 1 >= md.length || !md[i + 1].trim() || /^#|^- |^\d+\. /.test(md[i + 1])) { body.push({ p: para.join(" ") }); para = []; }
}

const today = new Date();
const dmy = `${String(today.getDate()).padStart(2, "0")}.${String(today.getMonth() + 1).padStart(2, "0")}.${today.getFullYear()}`;

// ---- title block
children.push(new Paragraph({ children: [new TextRun({ text: title, bold: true, size: 56, color: NAVY, font: FONT })], spacing: { before: 2400, after: 200 } }));
children.push(new Paragraph({ children: [new TextRun({ text: "User Guide", size: 32, color: NAVY, font: FONT })], spacing: { after: 400 } }));
children.push(new Paragraph({ children: [new TextRun({ text: subtitle.split(". ")[0] + ".", font: FONT })], spacing: { after: 120 } }));
children.push(new Paragraph({ children: [new TextRun({ text: `Version 0.5.0  |  ${dmy}`, font: FONT })], spacing: { after: 120 } }));
children.push(new Paragraph({ children: [new TextRun({ text: "nextE Asset Management SRL  |  CONFIDENTIAL", font: FONT, bold: true })], spacing: { after: 120 } }));
children.push(new Paragraph({ children: [new TextRun({ text: "Prepared for internal use by the nextE supply, trading and asset-management teams. Live application: https://nexte-esb.streamlit.app", font: FONT, size: 20, color: "6A6A6A" })], spacing: { after: 200 } }));
children.push(new Paragraph({ children: [new PageBreak()] }));
// ---- contents
children.push(new Paragraph({ text: "Contents", heading: HeadingLevel.HEADING_1 }));
toc.forEach((t, k) => children.push(new Paragraph({ children: [new TextRun({ text: `${k + 1}.  ${t.replace(/^\d+\.\s*/, "")}`, font: FONT })], spacing: { after: 60 } })));
children.push(new Paragraph({ children: [new PageBreak()] }));
// ---- body
for (const b of body) {
  if (b.h === 1) children.push(new Paragraph({ text: b.t, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } }));
  else if (b.h === 2) children.push(new Paragraph({ text: b.t, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } }));
  else if (b.b !== undefined) children.push(new Paragraph({ children: inline(b.b), numbering: { reference: "bullets", level: 0 }, spacing: { after: 80 } }));
  else if (b.n !== undefined) children.push(new Paragraph({ children: inline(b.n), numbering: { reference: "numbers", level: 0 }, spacing: { after: 80 } }));
  else children.push(new Paragraph({ children: inline(b.p), spacing: { after: 120 }, alignment: AlignmentType.JUSTIFIED }));
}

const doc = new Document({
  creator: "nextE Asset Management SRL",
  title: `${title} - User Guide`,
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: "0E1C2E" } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: NAVY, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ children: [new TextRun({ text: "nextE  |  Energy Supply Bid Management Tool  |  User Guide", size: 18, color: "6A6A6A", font: FONT })],
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C8C8C6" } } })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "Page ", size: 18, color: "6A6A6A", font: FONT }), new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "6A6A6A", font: FONT }),
      new TextRun({ text: "  |  CONFIDENTIAL · nextE", size: 18, color: "6A6A6A", font: FONT })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("written", out, buf.length); });
