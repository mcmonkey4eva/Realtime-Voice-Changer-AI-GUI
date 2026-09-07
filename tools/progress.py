import math

from i18n.i18n import I18nAuto


i18n = I18nAuto()


def should_report(index, total, max_updates=12):
    if total <= 0:
        return False
    if total <= max_updates:
        return True
    interval = max(1, math.ceil(total / max_updates))
    return index == 0 or index + 1 == total or (index + 1) % interval == 0


def batch_status(title, current, total, success, failed, latest="", failures=None):
    if total <= 0:
        state = i18n("Waiting for input")
    elif current >= total:
        state = i18n("Completed")
    else:
        state = i18n("Processing")
    lines = [
        "[%s]" % title,
        "%s: %s" % (i18n("Status"), state),
        "%s: %s/%s | %s: %s | %s: %s"
        % (
            i18n("Progress"),
            current,
            total,
            i18n("Success"),
            success,
            i18n("Failed"),
            failed,
        ),
    ]
    if latest:
        lines.append("%s: %s" % (i18n("Current"), latest))
    if failures:
        lines.append("%s: " % i18n("Failure records"))
        lines.extend(failures[-10:])
        if len(failures) > 10:
            lines.append(i18n("…Showing only the 10 most recent failures"))
    return "\n".join(lines)
