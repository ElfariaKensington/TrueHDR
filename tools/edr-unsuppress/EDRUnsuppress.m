// EDRUnsuppress: keeps full EDR headroom in Safari while another app is frontmost.
//
// When Safari stops being the active app, AppKit posts
// NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification. Stock WebKit's UI
// process (WebProcessPool and WebViewImpl observers) answers by clamping the headroom it sends to
// every web/GPU process to 1.6x (WebProcessPool::applyEDRSuppressionIfNeeded), which caps
// canvas, <img> and <video> alike (measured with dev/focus-test on Safari 27.3). The notification
// is advisory: apps that ignore it keep full headroom. This library drops that one notification
// inside Safari processes, so WebKit never starts suppressing. The matching "end" notification is
// left alone (harmless when nothing was suppressed).
//
// Loaded into AppKit apps by the FloatingSidebar loader (Plugins/ next to it); does nothing
// outside Safari (com.apple.Safari, Safari web apps com.apple.Safari.WebApp.*).
#import <Foundation/Foundation.h>
#import <objc/runtime.h>

static NSString *const kBeginSuppressing = @"NSApplicationShouldBeginSuppressingHighDynamicRangeContentNotification";

static IMP gPostNote, gPostNameObj, gPostNameObjInfo;

static void post_note(id self, SEL _cmd, NSNotification *n) {
    if ([n.name isEqualToString:kBeginSuppressing]) return;
    ((void (*)(id, SEL, NSNotification *))gPostNote)(self, _cmd, n);
}

static void post_name_obj(id self, SEL _cmd, NSNotificationName name, id obj) {
    if ([name isEqualToString:kBeginSuppressing]) return;
    ((void (*)(id, SEL, NSNotificationName, id))gPostNameObj)(self, _cmd, name, obj);
}

static void post_name_obj_info(id self, SEL _cmd, NSNotificationName name, id obj, NSDictionary *info) {
    if ([name isEqualToString:kBeginSuppressing]) return;
    ((void (*)(id, SEL, NSNotificationName, id, NSDictionary *))gPostNameObjInfo)(self, _cmd, name, obj, info);
}

static IMP swap(SEL sel, IMP repl) {
    Method m = class_getInstanceMethod([NSNotificationCenter class], sel);
    return m ? method_setImplementation(m, repl) : NULL;
}

__attribute__((constructor)) static void EDRUnsuppressInit(void) {
    @autoreleasepool {
        NSString *bid = NSBundle.mainBundle.bundleIdentifier ?: @"";
        BOOL force = getenv("EDR_UNSUPPRESS_ANY_APP") != NULL;  // tests only
        if (!force && !([bid isEqualToString:@"com.apple.Safari"] || [bid hasPrefix:@"com.apple.Safari.WebApp"])) return;
        gPostNote = swap(@selector(postNotification:), (IMP)post_note);
        gPostNameObj = swap(@selector(postNotificationName:object:), (IMP)post_name_obj);
        gPostNameObjInfo = swap(@selector(postNotificationName:object:userInfo:), (IMP)post_name_obj_info);
        NSLog(@"[EDRUnsuppress] active in %@ (pid %d)", bid, getpid());
    }
}
