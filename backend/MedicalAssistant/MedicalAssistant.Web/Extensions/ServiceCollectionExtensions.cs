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
        /// Registers application services for the Patient and Appointment modules (repositories + services).
        /// </summary>
        public static IServiceCollection AddPatientModule(this IServiceCollection services)
        {
            // Unit of Work + repositories
            services.AddScoped<IUnitOfWork, UnitOfWork>();
            services.AddScoped<IAppointmentRepository, AppointmentRepository>();
            services.AddScoped<ISessionRepository, SessionRepository>();
            services.AddScoped<IMessageRepository, MessageRepository>();

            // Services
            services.AddScoped<IPatientService, PatientService>();
            services.AddScoped<IAppointmentService, AppointmentService>();
            services.AddScoped<ISessionService, SessionService>();
            services.AddScoped<IMessageService, MessageService>();

            return services;
        }
    }
}
