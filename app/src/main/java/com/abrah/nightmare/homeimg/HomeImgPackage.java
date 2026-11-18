package com.abrah.nightmare.homeimg;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;

import java.io.*;

final class HomeImgPackage {
    interface Progress { void update(String file, long copied, long total) throws IOException; }
    private static final String MODEL = "homeimg-epoch49-v1";
    private static final String[] FILES = {
        "model_index.json", "tokenizer/tokenizer.json", "text_encoder/config.json",
        "text_encoder/diffusion_pytorch_model.safetensors", "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors", "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors"
    };

    static File installed(Context c) {
        String active = c.getSharedPreferences("homeimg", 0).getString("active", MODEL);
        return new File(c.getFilesDir(), active);
    }

    static synchronized void install(Context c, Uri tree, Progress progress) throws Exception {
        File previous = installed(c);
        String next = MODEL.equals(previous.getName()) ? MODEL + "-b" : MODEL;
        File stage = new File(c.getFilesDir(), next);
        remove(stage);
        if (!stage.mkdir()) throw new IOException("Cannot create checkpoint folder");
        boolean published = false;
        try {
            String root = DocumentsContract.getTreeDocumentId(tree);
            byte[] buffer = new byte[1024 * 1024];
            for (String path : FILES) {
                String id = root;
                for (String part : path.split("/")) id = child(c, tree, id, part);
                Uri uri = DocumentsContract.buildDocumentUriUsingTree(tree, id);
                long size = -1;
                try (Cursor cursor = c.getContentResolver().query(uri,
                        new String[]{DocumentsContract.Document.COLUMN_SIZE}, null, null, null)) {
                    if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) size = cursor.getLong(0);
                }
                File dest = new File(stage, path);
                File folder = dest.getParentFile();
                if (!folder.isDirectory() && !folder.mkdirs()) throw new IOException("Cannot create " + folder);
                long written = 0;
                try (InputStream in = c.getContentResolver().openInputStream(uri);
                     OutputStream out = new FileOutputStream(dest)) {
                    if (in == null) throw new IOException("Cannot open " + path);
                    int n;
                    while ((n = in.read(buffer)) != -1) {
                        out.write(buffer, 0, n);
                        written += n;
                        progress.update(path, written, size);
                    }
                }
                if (written == 0 || (size >= 0 && size != written)) throw new IOException("Incomplete file: " + path);
            }
            if (!c.getSharedPreferences("homeimg", 0).edit().putString("active", next).commit())
                throw new IOException("Cannot activate checkpoint");
            published = true;
            remove(previous);
        } finally {
            if (!published) remove(stage);
        }
    }

    private static String child(Context c,Uri tree,String id,String name) throws IOException {
        Uri children=DocumentsContract.buildChildDocumentsUriUsingTree(tree,id);
        String[] cols={DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME};
        try(Cursor cur=c.getContentResolver().query(children,cols,null,null,null)) {
            if(cur!=null) while(cur.moveToNext()) if(name.equals(cur.getString(1))) return cur.getString(0);
        }
        throw new IOException("Missing file or directory: "+name);
    }
    private static void remove(File file) throws IOException {
        if(!file.exists())return;
        if(file.isDirectory()){
            File[] children=file.listFiles();
            if(children==null)throw new IOException("Cannot inspect "+file);
            for(File child:children)remove(child);
        }
        if(!file.delete())throw new IOException("Cannot remove "+file);
    }
}
