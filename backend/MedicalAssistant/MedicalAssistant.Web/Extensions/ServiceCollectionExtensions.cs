using MedicalAssistant.Application.Services;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using Microsoft.Extensions.DependencyInjection;

namespace MedicalAssistant.Web.Extensions
{
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Registers all application services and repositories.
        /// </summary>
        public static IServiceCollection AddApplicationServices(this IServiceCollection services)
        {
            // ─── Unit of Work (registers all repositories internally) ──────
            services.AddScoped<IUnitOfWork, UnitOfWork>();

            // ─── Domain Services ──────────────────────────────────────────
            services.AddScoped<IPatientService,     PatientService>();
            services.AddScoped<IAppointmentService, AppointmentService>();
            services.AddScoped<IDoctorService,      DoctorService>();
            services.AddScoped<IReviewService,      ReviewService>();
            services.AddScoped<ISessionService,     SessionService>();
            services.AddScoped<IMessageService,     MessageService>();
            services.AddScoped<IAuthService,        AuthService>();
            services.AddScoped<IAdminService,       AdminService>();

            return services;
        }

        /// <summary>
        /// Legacy method - kept for backward compatibility.
        /// Prefer calling AddApplicationServices() instead.
        /// </summary>
        public static IServiceCollection AddPatientModule(this IServiceCollection services)
            => services.AddApplicationServices();
    }
}