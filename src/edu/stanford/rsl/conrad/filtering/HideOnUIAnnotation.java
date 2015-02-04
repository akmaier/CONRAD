package edu.stanford.rsl.conrad.filtering;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * By marking a class with this annotation, you can suppress it from appearing on the list of filters on the UI
 *
 * Example:
 * 
 * @HideOnUI
 * public class BilateralFiltering3DTool extends ImageFilteringTool { ... }
 */

@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface HideOnUIAnnotation {

}
