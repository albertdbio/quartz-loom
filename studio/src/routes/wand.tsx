/**
 * Legacy alias. Builds shipped before the Mochiverse rename point at `/wand`
 * (and `/api/**` redirects used to land here), so the path has to keep
 * resolving — an installed app cannot be migrated retroactively.
 */
export { default } from "./app"
