package com.abrah.nightmare.homeimg;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.format.DateUtils;
import android.widget.*;
import android.view.View;
import android.view.WindowInsets;
import android.graphics.Insets;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class HomeImgActivity extends Activity {
    private static final int PICK=1, SAVE=2;
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private volatile boolean stopped;
    private TextView status;
    private Chronometer elapsed;
    private boolean timingGeneration;
    private Button pick,run,save,probeButton;
    private EditText prompt,negative,steps,cfg,seed,timeShift;
    private ImageView image;
    private Bitmap result;
    private boolean busy;
    private Spinner accelerator;
    private String activeAccelerator;
    private SeekBar resolution;
    private String backend="CPU";
    private TextView backendDetails,stepTimes;
    private CheckBox gpuAttention;
    private void stepProfile(int step,long total,long projections,long attention,long mlp){
        String text=String.format(java.util.Locale.ROOT,
            "Step %d: %.1f s · accelerator projections %.1f s · GPU attention %.1f s · GPU MLP %.1f s%s",
            step,total/1000.,projections/1000.,attention/1000.,mlp/1000.,step==1?" (includes prompt)":"");
        runOnUiThread(()->{if(!stopped){stepTimes.setText(text);stepTimes.setVisibility(View.VISIBLE);}});
    }
    private void showGpuAttention(){
        if(resolution!=null && gpuAttention!=null)
            gpuAttention.setVisibility(accelerator.getSelectedItemPosition()==1?View.VISIBLE:View.GONE);
    }
    private void backendNotice(String reason){
        runOnUiThread(()->{
            if(!stopped){
                backendDetails.setText(reason);
                backendDetails.setVisibility(View.VISIBLE);
            }
        });
    }
    private LinearLayout layout;
    private native String probe(byte[] directory,boolean gpu);
    private native int[] generate(byte[] directory,byte[] prompt,byte[] negative,int resolution,int steps,float cfg,float shift,long seed,byte[] skelDirectory,boolean gpu,boolean gpuAttention);
    private static byte[] utf(String text){return text.getBytes(StandardCharsets.UTF_8);}
    private void backendSummary(int hits,int misses,long millis,long elapsed,long dispatches,int attentionHits,int attentionMisses,long attentionMs,int mlpHits,int mlpMisses,long mlpMs){
        backend="Generation "+elapsed+" ms · "+(hits+misses==0?"CPU":
                activeAccelerator+" projection calls "+millis+" ms · "+hits+" completed · "+misses+" CPU fallbacks")+(dispatches>0?" · "+(dispatches-1)+" Vulkan dispatches":"");
        if(attentionHits+attentionMisses>0)backend+=" · attention "+attentionMs+" ms ("+attentionHits+" GPU, "+attentionMisses+" fallback)";
        if(mlpHits+mlpMisses>0)backend+=" · MLP "+mlpMs+" ms ("+mlpHits+" GPU, "+mlpMisses+" fallback)";
    }
    private void report(String text){runOnUiThread(()->{if(!stopped)status.setText(text);});}
    private int completedSteps, totalSteps;
    private long lastProjectionReport;
    private void projectionStatus(int hits,int misses,boolean starting){
        if(completedSteps==totalSteps)return;
        long now=SystemClock.elapsedRealtime();
        if(hits+misses>1 && now-lastProjectionReport<500)return;
        lastProjectionReport=now;
        report("Step "+(completedSteps+1)+" / "+totalSteps+" · "+activeAccelerator
                +" · "+hits+" projections completed · "+misses+" on CPU"
                +(starting?" · dispatching":""));
    }
    private boolean progress(int done,int total){
        if(stopped)return false;
        completedSteps=done;
        totalSteps=total;
        report(done==0?"Preparing prompt and model · 0 / "+total:done==total?"Decoding image…":"Generating: "+done+" / "+total);
        return true;
    }
    @Override public void onCreate(Bundle state){
        super.onCreate(state);
        setTitle("HomeIMG");
        ScrollView scroll=new ScrollView(this);
        scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            Insets safe = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout()
                    | WindowInsets.Type.ime());
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom);
            return insets;
        });
        layout=new LinearLayout(this);layout.setOrientation(LinearLayout.VERTICAL);
        int padding=(int)(16*getResources().getDisplayMetrics().density);
        layout.setPadding(padding,padding,padding,padding);scroll.addView(layout);setContentView(scroll);

        pick=button("Import HomeIMG checkpoint",v->startActivityForResult(new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE),PICK));
        prompt=field("Prompt","");
        negative=field("Negative prompt (optional)","");
        steps=field("Steps","50");cfg=field("Guidance","3");seed=field("Seed","49");
        timeShift=field("Time shift","");
        timeShift.setHint("Auto (checkpoint)");

        boolean available=false;
        try{available=android.os.Build.SOC_MODEL.toUpperCase(java.util.Locale.ROOT).startsWith("SM8550")
                && java.util.Arrays.asList(getAssets().list("homeimg-htp")).contains("libggml-htp-v73.so");}
        catch(IOException ignored){}
        TextView backendLabel=new TextView(this);
        backendLabel.setText("HomeIMG backend");
        layout.addView(backendLabel);
        accelerator=new Spinner(this);
        String[] choices=available ? new String[]{"CPU","Vulkan GPU","HTP v73"} : new String[]{"CPU","Vulkan GPU"};
        ArrayAdapter<String> adapter=new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,choices);
        accelerator.setAdapter(adapter);
        layout.addView(accelerator);
        gpuAttention=new CheckBox(this);
        gpuAttention.setText("Run self-attention on GPU");
        gpuAttention.setChecked(true);
        gpuAttention.setVisibility(View.GONE);
        layout.addView(gpuAttention);
        accelerator.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener(){
            public void onItemSelected(AdapterView<?> parent,View v,int position,long id){showGpuAttention();}
            public void onNothingSelected(AdapterView<?> parent){showGpuAttention();}
        });
        probeButton=button("Test selected backend",v->{
            int choice=accelerator.getSelectedItemPosition();
            if(choice==0){report("CPU selected");return;}
            submit(()->{
                System.loadLibrary("homeimg_cpu");
                report(probe(utf(choice==2?skelDir():""),choice==1));
            });
        });
        TextView sizeLabel = new TextView(this);
        layout.addView(sizeLabel);
        resolution = new SeekBar(this);
        resolution.setMax((1024 - 256) / 16);
        layout.addView(resolution);
        resolution.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar bar, int progress, boolean fromUser) {
                showGpuAttention();
                int size = selectedSize();
                sizeLabel.setText("Resolution: " + size + " × " + size);
                bar.setContentDescription(sizeLabel.getText());
            }
            public void onStartTrackingTouch(SeekBar bar) {}
            public void onStopTrackingTouch(SeekBar bar) {}
        });
        sizeLabel.setText("Resolution: 256 × 256");
        resolution.setContentDescription(sizeLabel.getText());
        run=button("Generate",v->startGeneration());
        elapsed=new Chronometer(this);
        elapsed.setFormat("Elapsed %s");
        elapsed.setVisibility(View.GONE);
        layout.addView(elapsed);
        status=new TextView(this);status.setTextIsSelectable(true);layout.addView(status);
        stepTimes=new TextView(this);
        stepTimes.setVisibility(View.GONE);
        layout.addView(stepTimes);
        backendDetails=new TextView(this);
        backendDetails.setTextIsSelectable(true);
        backendDetails.setVisibility(View.GONE);
        layout.addView(backendDetails);
        image=new ImageView(this);image.setAdjustViewBounds(true);layout.addView(image);
        save=button("Save PNG",v->startActivityForResult(new Intent(Intent.ACTION_CREATE_DOCUMENT)
            .setType("image/png").putExtra(Intent.EXTRA_TITLE,"HomeIMG.png"),SAVE));
        status.setText(HomeImgPackage.installed(this).isDirectory()?"HomeIMG ready":"Select the folder containing model_index.json");
        controls();
    }
    private Button button(String text,View.OnClickListener click){
        Button b=new Button(this);b.setText(text);b.setOnClickListener(click);layout.addView(b);return b;
    }
    private EditText field(String label,String value){
        TextView title=new TextView(this);title.setText(label);layout.addView(title);
        EditText e=new EditText(this);e.setSingleLine();e.setText(value);layout.addView(e);return e;
    }
    private void controls(){
        boolean installed=HomeImgPackage.installed(this).isDirectory();
        if(probeButton!=null)probeButton.setEnabled(!busy);
        gpuAttention.setEnabled(!busy);accelerator.setEnabled(!busy);resolution.setEnabled(!busy);pick.setEnabled(!busy);run.setEnabled(!busy && installed);save.setEnabled(!busy && result!=null);
    }
    interface Job { void run() throws Exception; }
    private void submit(Job job){
        if(busy)return;
        busy=true;controls();
        worker.execute(()->{
            try{job.run();}catch(Exception|LinkageError e){report("HomeIMG: "+e.getMessage());}
            finally{runOnUiThread(()->{if(!stopped){finishTimer();busy=false;controls();}});}
        });
    }
    @Override protected void onActivityResult(int request,int code,Intent data){
        super.onActivityResult(request,code,data);
        if(code!=RESULT_OK||data==null||data.getData()==null)return;
        Uri uri=data.getData();
        if(request==PICK)submit(()->{
            long[] last={0};
            HomeImgPackage.install(this,uri,(path,n,total)->{
                if(stopped)throw new IOException("Import interrupted");
                long now=SystemClock.elapsedRealtime();
                if(now-last[0]>=400){report("Importing "+path+(total>0?" · "+(100*n/total)+"%":""));last[0]=now;}
            });
            report("HomeIMG ready");
        });
        if(request==SAVE && result!=null)submit(()->{
            try(OutputStream out=getContentResolver().openOutputStream(uri,"wt")){
                if(out==null || !result.compress(Bitmap.CompressFormat.PNG,100,out))throw new IOException("Cannot save PNG");
            }
            report("PNG saved");
        });
    }
    private int selectedSize() { return 256 + 16 * resolution.getProgress(); }
    private void finishTimer(){
        if(!timingGeneration)return;
        elapsed.stop();
        long seconds=(SystemClock.elapsedRealtime()-elapsed.getBase())/1000;
        elapsed.setText("Elapsed "+DateUtils.formatElapsedTime(seconds));
        timingGeneration=false;
    }
    private void startGeneration(){
        if(busy)return;
        try{
            int n=Integer.parseInt(steps.getText().toString());
            float g=Float.parseFloat(cfg.getText().toString());long s=Long.parseLong(seed.getText().toString());
            if(n<1||n>1000||!Float.isFinite(g))throw new IllegalArgumentException("Check steps and guidance");
            byte[] positive=utf(prompt.getText().toString()), neg=utf(negative.getText().toString());
            String enteredShift=timeShift.getText().toString().trim().replace(',', '.');
            float shift=0f;
            if (!enteredShift.isEmpty()) {
                try { shift=Float.parseFloat(enteredShift); }
                catch (NumberFormatException ignored) { timeShift.setError("Enter a positive number"); return; }
                if (!Float.isFinite(shift) || shift<=0f) {
                    timeShift.setError("Enter a finite positive number");
                    return;
                }
            }
            final float requestedShift=shift;
            int size=selectedSize();
            int choice=accelerator.getSelectedItemPosition();
            activeAccelerator=accelerator.getSelectedItem().toString();
            backendDetails.setVisibility(View.GONE);
            stepTimes.setVisibility(View.GONE);
            boolean useAttention=gpuAttention.isChecked() && choice==1;
            elapsed.setBase(SystemClock.elapsedRealtime());
            elapsed.setVisibility(View.VISIBLE);
            timingGeneration=true;
            elapsed.start();
            submit(()->{
                System.loadLibrary("homeimg_cpu");
                int[] pixels=generate(utf(HomeImgPackage.installed(this).getPath()),positive,neg,size,n,g,requestedShift,s,utf(choice==2?skelDir():""),choice==1,useAttention);
                if(stopped||pixels==null)return;
                Bitmap b=Bitmap.createBitmap(pixels,size,size,Bitmap.Config.ARGB_8888);
                runOnUiThread(()->{if(!stopped){result=b;image.setImageBitmap(b);status.setText("Image ready · "+backend);}});
            });
        }catch(IllegalArgumentException e){report("Check steps, guidance and seed");}
    }
    private String skelDir() throws IOException {
        File directory=new File(getFilesDir(),"homeimg-dsp");
        if(!directory.isDirectory() && !directory.mkdir())throw new IOException("Cannot prepare HTP library");
        File temp=File.createTempFile("skel",".tmp",directory);
        String name="libggml-htp-v73.so";
        try {
            try(InputStream in=getAssets().open("homeimg-htp/"+name);OutputStream out=new FileOutputStream(temp)){
                byte[] buf=new byte[65536];int n;
                while((n=in.read(buf))!=-1)out.write(buf,0,n);
            }
            if(!temp.renameTo(new File(directory,name)))throw new IOException("Cannot install HTP library");
        }finally{if(temp.exists())temp.delete();}
        return directory.getAbsolutePath();
    }
    @Override protected void onDestroy(){stopped=true;finishTimer();worker.shutdown();super.onDestroy();}
}
